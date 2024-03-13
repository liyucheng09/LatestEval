import datasets
from glob import glob
import os
import json

if __name__ == '__main__':
    files = glob('/vol/research/lyc/wikitext_alltime/wiki/*.json')
    hf_token = os.environ['HF_TOKEN']

    for file in files:
        
        all_articles = []

        time = os.path.basename(file).strip('.json')
        year = int(time.split('-')[0])
        month = int(time.split('-')[1])

        time_stamp = f'{year}-{month:02d}'
        if time_stamp not in ['2024-01', '2024-02']:
            continue
        print(f"Processing {time_stamp}")

        with open(file) as f:
            data = json.load(f)
        
        for title, article in data.items():
            article['time'] = time_stamp
            all_articles.append(article)
        
        ds = datasets.Dataset.from_list(all_articles)
        ds.push_to_hub(f"RealTimeData/wikitext_alltime", config_name=time_stamp, token=hf_token)
