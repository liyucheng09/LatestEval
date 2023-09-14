import requests
import traceback
import mwparserfromhell
import datetime
import datasets
from huggingface_hub import create_branch, create_tag, RepoCard
import os

WIKI_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

def parse_to_plain_text(wikitext):
    parsed = mwparserfromhell.parse(wikitext)
    return parsed.strip_code()

def fetch_recent_changes(from_date, to_date = '2023-08-01T00:00:00', limit = 2000, continue_token = None):
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
    if continue_token is not None:
        params['rccontinue'] = continue_token
    req = requests.Request('GET', WIKI_API_ENDPOINT, params=params).prepare()
    response = requests.get(WIKI_API_ENDPOINT, params=params).json()
    
    # Check if the response contains the expected data
    if 'query' in response and 'recentchanges' in response['query']:
        results = [entry['title'] for entry in response['query']['recentchanges']]
        num_results = len(results)
        if num_results < limit:
            continue_token = response['continue']['rccontinue'] if 'continue' in response else None
            if continue_token is not None:
                results += fetch_recent_changes(from_date, to_date, limit - num_results, continue_token)
        return results
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

    hf_token = os.environ['HF_TOKEN']

    today = datetime.datetime.now()
    one_week_ago = today - datetime.timedelta(days=7)
    two_week_ago = today - datetime.timedelta(days=14)

    new_articles = fetch_recent_changes(two_week_ago.isoformat(), one_week_ago.isoformat(), limit=8000)
    print(f'Num new articles: {len(new_articles)}')

    articles = []
    for article in new_articles:
        content = fetch_content(article)
        if content is not None:
            articles.append(content)
    
    ds = datasets.Dataset.from_dict({key: [article[key] for article in articles] for key in articles[0].keys()})
    print(ds)
    try:
        create_branch("RealTimeData/wikitext_latest", branch=one_week_ago.date().isoformat(), repo_type="dataset", token=hf_token)
    except:
        traceback.print_exc()
        
    ds.push_to_hub('RealTimeData/wikitext_latest', branch='main', token=hf_token)
    ds.push_to_hub('RealTimeData/wikitext_latest', branch=one_week_ago.date().isoformat(), token=hf_token)

    text = f"""
# Latest Wikitext

You could always access the latest Wikipedia texts via this dataset.

We update the dataset weekly, on every Sunday. So the dataset always provides the latest Wikipedia texts from the last week.

The current dataset on main branch contains the latest wikipedia texts created from {two_week_ago.date().isoformat()} to {one_week_ago.date().isoformat()}.

The data collection is conducted on {today.date().isoformat()}.

Use the dataset via:
```
ds = datasets.load_dataset('RealTimeData/wikitext_latest')
```

# Previsou versions

You could access previous versions by requesting different branches.

For example, you could find the 2023-08-12 version via:
```
ds = datasets.load_dataset('RealTimeData/wikitext_latest', revision = '2023-08-12')
```

Check all available versions by clicking the "Files and versions" button on the top bar.
"""
    card = RepoCard(text)
    card.push_to_hub('RealTimeData/wikitext_latest', repo_type='dataset', token=hf_token)