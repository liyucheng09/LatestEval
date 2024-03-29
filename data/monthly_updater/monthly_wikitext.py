import requests
import traceback
import mwparserfromhell
import datetime
import os
import datasets

import sys
import json
import time

from tqdm import tqdm

WIKI_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

def parse_to_plain_text(wikitext):
    parsed = mwparserfromhell.parse(wikitext)
    return parsed.strip_code()

def fetch_content(title, date=None):
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvlimit": "1",
    }
    if date: params["rvstart"] = date
    try:
        response = requests.get(WIKI_API_ENDPOINT, params=params)
        response.raise_for_status()  # Will raise an error if the HTTP request returned an unsuccessful status code
        data = response.json()
        if 'error' in data:
            print(f"Error fetching content for {title}: {data['error']['info']}")
            return None

        page = next(iter(data['query']['pages'].values()))
        if 'revisions' not in page:
            print(f"No revisions found for {title}")
            return None
        content = page['revisions'][0]['*']
        
        # Check if the content is a redirect and skip if true
        if content.lower().startswith("#redirect"):
            print(f"{title} is a redirect page.")
            return None
        text = parse_to_plain_text(content)
        if len(text.split(' ')) < 300:
            print(f"{title} is less than 300 words.")
            return None
        
        return {
            "title": page['title'],
            "text": text,
            "pageid": page['pageid'],
        }

    except Exception as e:
        print(f"An error occurred while fetching content for {title}: {str(e)}")
        traceback.print_exc()  # This will print the full traceback

    return None

if __name__ == "__main__":
    today = datetime.datetime.today()
    year = today.year
    month = today.month

    hf_token = os.environ['HF_TOKEN']

    start_time = datetime.datetime(year, month, 1)
    end_time = today

    print(f'Fetching wiki articles from {start_time.isoformat()} to {end_time.isoformat()}')

    with open('./data/squad_wiki_title.text') as f:
        titles = [line.strip() for line in f.readlines()]
    historical_contents = [fetch_content(title, end_time) for title in tqdm(titles)]
    historical_contents = [content for content in historical_contents if content is not None]
    historical_to_save = {title: content for title, content in zip(titles, historical_contents)}

    save_file = f'{year}-{month}.json'
    with open(save_file, 'w') as f:
        json.dump(historical_to_save, f, ensure_ascii=False)
    print(f'Saved {len(historical_contents)} articles to {save_file}')

    from huggingface_hub import hf_hub_download, RepoCard, upload_file

    with open(save_file) as f:
        data = json.load(f)
    
    all_articles = []
    for title, article in data.items():
        article['time'] = f'{year}-{month:02d}'
        all_articles.append(article)
    
    ds = datasets.Dataset.from_list(all_articles)
    ds.push_to_hub(f"RealTimeData/wikitext_alltime", config_name=f'{year}-{month:02d}', token=hf_token)

    text = f"""
# Wikitext for All Times

You could find 491 selected wiki articles every month from 2017-01 to {year}-{month:02d}.

Use this to download wiki articles during a specific month:
```
ds = datasets.load_dataset('RealTimeData/wikitext_alltime', '2017-08')
```

The time stamp follows the format of "YYYY-MM".

# An example

```
> ds = datasets.load_dataset('RealTimeData/wikitext_alltime', '2023-10', split='train')
> ds[0]
```
"""
    card = RepoCard(text)
    card.push_to_hub('RealTimeData/wikitext_alltime', repo_type='dataset', token=hf_token)