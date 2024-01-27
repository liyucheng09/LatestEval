import datetime
import yt_dlp
from googleapiclient.discovery import build
import os
import sys
import re
from glob import glob
import datasets
import time
import random

def get_popular_videos(youtube, start_date, end_date, max_results=30):
    published_after = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    published_before = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    request = youtube.search().list(
        part="snippet",
        maxResults=max_results,
        order="viewCount",
        publishedAfter=published_after,
        publishedBefore=published_before,
        type="video",
    )
    response = request.execute()
    videos_ids = [item['id']['videoId'] for item in response['items']]
    return videos_ids

def parse_duration(duration_string):
    hours = re.search(r'(\d+)H', duration_string)
    minutes = re.search(r'(\d+)M', duration_string)
    seconds = re.search(r'(\d+)S', duration_string)

    hours = int(hours.group(1)) if hours else 0
    minutes = int(minutes.group(1)) if minutes else 0
    seconds = int(seconds.group(1)) if seconds else 0

    return hours * 3600 + minutes * 60 + seconds

def filter_too_long_video(youtube, video_ids, max_duration=600, max_results=30):
    request = youtube.videos().list(
        part="contentDetails",
        id=','.join(video_ids),
    )
    response = request.execute()

    final_videos = []
    for item in response['items']:
        duration = parse_duration(item['contentDetails']['duration'])
        if duration <= max_duration:
            final_videos.append(item['id'])
        if len(final_videos) >= max_results:
            break
    return final_videos

def download_audio(video_id, save_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'flac',
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ar', '16000'  # Set audio sample rate to 16 kHz
        ],
        'outtmpl': os.path.join(save_path, '%(id)s.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f'http://www.youtube.com/watch?v={video_id}'])

if __name__ == '__main__':

    month, save_path, = sys.argv[1:]
    month = int(month) + 1

    videos_per_month = 5
    api_key = os.environ['YOUTUBE_API_KEY']
    youtube = build('youtube', 'v3', developerKey=api_key)

    time_stamps = [f'{year}-{month:02d}' for year in range(2020, 2024)]
    # time_stamps = [f'{year}-{month:02d}' for year in range(2020, 2024) for month in range(1, 13)]

    for time_stamp in time_stamps:
        files = glob(os.path.join(save_path, time_stamp, '*.flac'))
        print(f"Start {time_stamp}...")

        if not len(files) >= videos_per_month:
            year, month = time_stamp.split('-')

            start_date = datetime.date(int(year), int(month), 1)
            end_of_month = datetime.date(int(year), int(month), 28)

            video_ids = get_popular_videos(youtube, start_date, end_of_month, max_results=50)
            video_ids = filter_too_long_video(youtube, video_ids, max_duration=600, max_results=videos_per_month)
            for video in video_ids:
                download_audio(video, os.path.join(save_path, time_stamp))
            
            print(f"Downloaded {len(video_ids)} videos in {time_stamp}")

        files = glob(os.path.join(save_path, time_stamp, '*.flac'))
        random.shuffle(files)
        files = files[:videos_per_month]
        ds = datasets.Dataset.from_dict({
            'audio': files,
            'name': files,
            'time': [time_stamp] * len(files),
        }).cast_column('audio', datasets.Audio())

        try:
            ds.push_to_hub('RealTimeData/audio_alltime', config_name=time_stamp, token=os.environ['HF_TOKEN'])
        except:
            print(f"Failed to push {time_stamp}")

        print(f"Finished {time_stamp}.")
