import requests
import mwparserfromhell
import json
import os
from lyc.utils import self_info
from transformers import LlamaForCausalLM, LlamaTokenizerFast
import sys
import torch
from tqdm import tqdm
import traceback

WIKI_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

def fetch_recent_changes(from_date, to_date = '2023-08-01T00:00:00'):
    params = {
        "action": "query",
        "format": "json",
        "list": "recentchanges",
        "rcstart": to_date,  # starting from the newer date
        "rcend": from_date,  # ending at the older date
        "rctype": "new",
        "rcnamespace": "0",
        "rclimit": "500",
        "rcprop": "title|timestamp"
    }
    req = requests.Request('GET', WIKI_API_ENDPOINT, params=params).prepare()
    response = requests.get(WIKI_API_ENDPOINT, params=params).json()
    
    # Check if the response contains the expected data
    if 'query' in response and 'recentchanges' in response['query']:
        return [entry['title'] for entry in response['query']['recentchanges']]
    else:
        return []

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
        if content.startswith("#REDIRECT"):
            print(f"{title} is a redirect page.")
            return None
        return content

    except Exception as e:
        print(f"An error occurred while fetching content for {title}: {str(e)}")
        traceback.print_exc()  # This will print the full traceback

    return None

def parse_to_plain_text(wikitext):
    parsed = mwparserfromhell.parse(wikitext)
    return parsed.strip_code()

def select_token_window(text, token_count=1000):
    tokens = text.split()
    if len(tokens) <= token_count:
        return text
    return ' '.join(tokens[:token_count])

def fetch_latest_and_historical_wiki_pages(cache_dir = ''):
    # 1. Fetch the latest created pages from July 2023 and their content.
    recent_wiki_path = os.path.join(cache_dir, 'recent_wiki_pages.json')
    if not os.path.exists(recent_wiki_path):
        recent_titles = fetch_recent_changes("2023-07-01T00:00:00Z")
        recent_contents = [fetch_content(title) for title in tqdm(recent_titles)]

        data_to_save = {title: content for title, content in zip(recent_titles, recent_contents)}
        with open(recent_wiki_path, 'w') as file:
            json.dump(data_to_save, file, ensure_ascii=False, indent=4)
    else:
        with open(recent_wiki_path) as file:
            data_to_save = json.load(file)
        recent_titles = list(data_to_save.keys())
        recent_contents = list(data_to_save.values())

    # 2. Fetch a historical version of a specific title from July 2022.
    historical_wiki_path = os.path.join(cache_dir, 'historical_wiki_pages.json')
    if not os.path.exists(historical_wiki_path):
        with open(os.path.join(cache_dir, 'data/squad_wiki_title.text')) as f:
            titles = [line.strip() for line in f.readlines()]
        historical_contents = [fetch_content(title, "2022-07-01T00:00:00Z") for title in tqdm(titles)]
        historical_to_save = {title: content for title, content in zip(titles, historical_contents)}
        with open(historical_wiki_path, 'w') as file:
            json.dump(historical_to_save, file, ensure_ascii=False, indent=4)
    else:
        with open(historical_wiki_path) as file:
            historical_to_save = json.load(file)
        historical_titles = list(historical_to_save.keys())
        historical_contents = list(historical_to_save.values())

    # 3. Parse the content to plain text.
    plain_texts_recent = [parse_to_plain_text(content) for content in recent_contents]
    plain_texts_historical = [parse_to_plain_text(content) for content in historical_contents]

    # 4. Select a 1000-token window from the text.
    selected_windows_recent = [select_token_window(text) for text in plain_texts_recent]
    selected_windows_historical = [select_token_window(text) for text in plain_texts_historical]

    return selected_windows_recent, selected_windows_historical

if __name__ == "__main__":
    cwd, model_name = sys.argv[1:]
    recent_snippets, historical_snippets = fetch_latest_and_historical_wiki_pages(cache_dir=cwd)
    
    recent_snippets = recent_snippets[:10]
    historical_snippets = historical_snippets[:10]
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)

    recent_info = []
    for recent in recent_snippets:
        tokens, info = self_info(recent, model, tokenizer)
        print(recent, info)
        recent_info.append(sum(info))
    
    historical_info = []
    for historical in historical_snippets:
        tokens, info = self_info(historical, model, tokenizer)
        print(historical, info)
        historical_info.append(sum(info))
    
    print('=====================')
    print(f'Model: {model_name}')
    print(f'Average self-info of recent snippets: {sum(recent_info)}, Average self-info of historical snippets: {sum(historical_info)}')