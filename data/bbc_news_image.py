import datasets
import requests
import sys
import os
from mimetypes import guess_extension
from PIL import Image
from time import sleep
import pandas as pd
import random

def download_image(url, save_path, headers, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                extension = guess_extension(response.headers['content-type']) or '.jpg'
                save_path = os.path.join(save_path, f"{i}{extension}")
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return save_path
            else:
                retries += 1
                sleep(1)  # Wait for 1 second before retrying
        except Exception as e:
            print(f"An error occurred: {e}")
            retries += 1
            sleep(1)
    return None

if __name__ == "__main__":

    hf_token = os.environ['HF_TOKEN']

    month, save_path, = sys.argv[1:]
    month = int(month) + 1

    header = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.114 Safari/537.36"
    }

    all_times = datasets.get_dataset_config_names('RealTimeData/bbc_news_alltime')

    # find times in the year
    times = ['2024-01', '2024-02']
    # for t in all_times:
    #     if t not in ['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12', '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12', '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-06', '2022-07', '2022-08', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-06', '2023-07', '2023-08', '2023-11', '2023-12']:
    #     # if f'{month:02d}' == t[-2:] and t not in ['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12', '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12', '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-06', '2022-07', '2022-08', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-06', '2023-07', '2023-08', '2023-11', '2023-12']:
    #         times.append(t)

    # load image links
    for time_stamp in times:
        print(f"Processing {time_stamp}")
        year = int(time_stamp.split('-')[0])

        save_path = os.path.join(save_path, str(year), str(month), 'images')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        sleep(random.randint(1, 10))
        ds = datasets.load_dataset('RealTimeData/bbc_news_alltime', time_stamp, split='train')
        urls = ds['top_image']
        article_links = ds['link']
        titles = ds['title']
        print(f"Found {len(urls)} links in {time_stamp}")

        all_images = []
        links_and_files = []
        for i, (url, link, title) in enumerate(zip(urls, article_links, titles)):
            if link is None:
                all_images.append({'url': url, 'img': None, 'title': title})
                continue
            
            # download image
            downloaded_image = download_image(url, save_path, header)
            if downloaded_image is None:
                print(f"Failed to download image {url}")
                all_images.append({'url': url, 'img': None, 'title': title})
                continue

            # load image
            img = Image.open(downloaded_image)
            file = os.path.basename(downloaded_image)
            all_images.append({'url': url, 'img': img, 'title': title})
            links_and_files.append({'link': link, 'file': file})

        # save links and files to metadata.csv
        df = pd.DataFrame(links_and_files)
        df.to_csv(os.path.join(save_path, 'metadata.csv'))

        # Huggingface datasets
        ds = datasets.Dataset.from_list(all_images)
        ds.push_to_hub(f"RealTimeData/bbc_images_alltime", config_name=time_stamp, token=hf_token)
        print(f"{time_stamp} done")
