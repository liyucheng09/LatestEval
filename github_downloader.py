import requests
import base64
from datetime import datetime, timedelta
import os
import json
from tqdm import tqdm
from huggingface_hub import create_branch, create_tag, RepoCard

github_token = os.environ['Github_Token']
headers = {'Authorization': f'token {github_token}'}

hf_token = os.environ['HF_TOKEN']

today = datetime.now()
start_date = today - timedelta(weeks=2)
start_date_str = start_date.strftime("%Y-%m-%d")

end_date = start_date + timedelta(days=7)
end_date_str = end_date.strftime("%Y-%m-%d")

out_path = f"dataset/github/{start_date_str}"
if not os.path.exists(out_path):
    os.makedirs(out_path)

def load_checkpoint():
    try:
        with open(f'{start_date_str}_checkpoint.json', 'r') as f:
            checkpoint = json.load(f)
            return checkpoint.get('page', 1), checkpoint.get('last_repo_index', 0)
    except FileNotFoundError:
        return 1, 0

def save_checkpoint(page, last_repo_index):
    with open(f'{start_date_str}_checkpoint.json', 'w') as f:
        json.dump({'page': page, 'last_repo_index': last_repo_index}, f)

page, last_repo_index = load_checkpoint()

while True:
    response = requests.get(f'https://api.github.com/search/repositories?q=created:{start_date_str}..{end_date_str}&sort=stars&order=desc&per_page=100&page={page}', headers=headers)
    data = response.json()

    if 'items' not in data:
        break
    if not data['items']:
        break
    for repo in tqdm(data['items'][last_repo_index:]):
        owner = repo['owner']['login']
        repo_name = repo['name']
        
        full_name = repo['full_name']
        url = repo['html_url']
        description = repo['description']
        stars = repo['stargazers_count']
        forks = repo['forks_count']

        response = requests.get(f'https://api.github.com/repos/{owner}/{repo_name}/readme', headers=headers)
        readme_data = response.json()

        if 'content' in readme_data:
            readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
            # print(f"Repository {repo_name} README content:")
            # print(readme_content)
            with open(f"{out_path}/{full_name.replace('/', '_')}_README.md", 'w') as f:
                json.dump({'full_name': full_name, 'url': url, 'description': description, 'readme': readme_content, 'stars': stars, 'forks': forks}, f, ensure_ascii=False)
        else:
            print(f"Repository {repo_name} doesn't have a README.")
        
        last_repo_index += 1
        save_checkpoint(page, last_repo_index)

    page += 1
    last_repo_index = 0
    save_checkpoint(page, last_repo_index)

import datasets
from glob import glob
files = glob(f'{out_path}/*.md')
ds = datasets.load_dataset('json', data_files = files, split = 'train')

create_branch("RealTimeData/github_latest", branch=start_date_str, repo_type="dataset", token=hf_token)
ds.push_to_hub("RealTimeData/github_latest", token=hf_token, branch='main')
ds.push_to_hub("RealTimeData/github_latest", token=hf_token, branch=start_date_str)

text = f"""
# Latest GitHub Repositories

You could always access the latest Github repos via this dataset.

We update the dataset weekly, on every Sunday. So the dataset always provides the latest Github repos from the last week.

The current dataset on main branch contains the latest Github Repos submitted from {start_date_str} to {end_date_str}.

The data collection is conducted on {today.date.isoformat()}.

Use the dataset via:
```
ds = datasets.load_dataset('RealTimeData/github_latest')
```

# Previsou versions

You could access previous versions by requesting different branches.

For example, you could find the 2023-08-06 version via:
```
ds = datasets.load_dataset('RealTimeData/github_latest', revision = '2023-08-06')
```

Check all available versions by clicking the "Files and versions" button on the top bar.
"""
card = RepoCard(text)
card.push_to_hub('RealTimeData/github_latest', repo_type='dataset', token=hf_token)