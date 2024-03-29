from git import Repo
import datetime
import os
import sys
import multiprocessing
import difflib
import json
import itertools
import shutil

def clone_repo(repo_url, local_path, overwrite=False, since=None):
    if os.path.exists(local_path):
        if overwrite:
            shutil.rmtree(local_path)
        else:
            print(f"Repo {local_path} already exists")
            return
    Repo.clone_from(repo_url, local_path, multi_options=[f'--shallow-since={since}'] if since is not None else None)

def get_file_content(commit, file_path):
    # Retrieves the file content for a given commit
    blob = commit.tree / file_path
    return blob.data_stream.read().decode('utf-8', errors='ignore')

def compare_files(repo, diff_item, start_commit, end_commit, code_extensions):
    file_path = diff_item.b_path
    _, ext = os.path.splitext(file_path)
    if ext not in code_extensions:
        # print(f"Skipping {file_path} because it is not a code file")
        return None

    # If the change type is added, we append it anyway
    # If the change type is modified, we only append it if the file was significantly changed (more than 50% of its lines were changed)
    if diff_item.change_type == 'M':
        a_content = get_file_content(start_commit, diff_item.a_path)
        b_content = get_file_content(end_commit, diff_item.b_path)

        # Use difflib to compare contents
        diff = difflib.unified_diff(
            a_content.splitlines(keepends=True),
            b_content.splitlines(keepends=True),
            fromfile=diff_item.a_path,
            tofile=diff_item.b_path
        )

        # Count the number of lines added
        changes = sum(1 for line in diff if line.startswith('+') and not line.startswith('++'))

        # if the file was not significantly changed, skip it. we consider a file significantly changed if more than 50% of its lines were changed
        if changes < 0.5 * len(a_content.splitlines()):
            return None
    elif diff_item.change_type == 'A':
        b_content = get_file_content(end_commit, diff_item.b_path)
        changes = len(b_content.splitlines())
    elif diff_item.change_type == 'R':
        # skip renamed files
        return None
    else:
        print(diff_item.change_type)
        return None

    return {
        'file_path': file_path,
        'num_changed_lines': changes,
        'code': b_content,
    }

def get_monthly_diff_file_objects(local_path, start_date, end_date, code_extensions):
    repo = Repo(local_path)
    repo_name = local_path.split('/')[-1]
    file_changes = []

    try:
        start_commit = next(repo.iter_commits(until=start_date))
        end_commit = next(repo.iter_commits(until=end_date))
    except StopIteration:
        print(f"Repo {local_path} has no commits in {start_date} - {end_date}")
        return file_changes

    start_commit_date = datetime.datetime.fromtimestamp(start_commit.committed_date).date()
    end_commit_date = datetime.datetime.fromtimestamp(end_commit.committed_date).date()
    end_commit_date_str = end_commit_date.strftime("%Y-%m-%d")

    if start_commit_date > end_date or end_commit_date < start_date:
        # print(f"Repo {local_path} has no commits in the given time range")
        return file_changes

    diff_index = start_commit.diff(end_commit, **{'find_renames=50%': True, 'insert_kwargs_after': '-r'})

    for diff_item in diff_index.iter_change_type('M'):
        result = compare_files(repo, diff_item, start_commit, end_commit, code_extensions)
        if result:
            result['repo_name'] = repo_name
            result['commit_date'] = end_commit_date_str
            result['sha'] = end_commit.hexsha
            file_changes.append(result)
    
    for diff_item in diff_index.iter_change_type('A'):
        result = compare_files(repo, diff_item, start_commit, end_commit, code_extensions)
        if result:
            result['repo_name'] = repo_name
            result['commit_date'] = end_commit_date_str
            result['sha'] = end_commit.hexsha
            file_changes.append(result)

    # Ranking files by the extent of added lines
    ranked_files = sorted(file_changes, key=lambda x: x['num_changed_lines'], reverse=True)
    # print(f"Total {len(ranked_files)} files changed")
    return ranked_files

def main(time_stamp, local_repo, save_path):
    year, month = time_stamp.split('-')
    first_day = datetime.date(int(year), int(month), 1)
    last_day = datetime.date(int(year), int(month), 28)
    
    repo_name = local_repo.split('/')[-1]
    # print(f"Processing {repo_name} at {time_stamp}")

    save_path = os.path.join(save_path, time_stamp, repo_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ranked_files = get_monthly_diff_file_objects(local_repo, first_day, last_day, code_extensions)
    for index, file in enumerate(ranked_files[:50]):
        save_file_path = os.path.join(save_path, f"{index}.json")
        with open(save_file_path, 'w') as f:
            json.dump(file, f, ensure_ascii=False, indent=2)
    return (time_stamp, repo_name, len(ranked_files))
    # print(f"Saved to {save_path}")

if __name__ == '__main__':
    today = datetime.date.today()
    year = today.year
    month = today.month

    time_stamp = f'{year}-{month:02d}'
    first_day_string = f'{year}-{month:02d}-01'

    repo_path, save_dir = 'repos/', 'code_data/'
    repo_list = 'data/code_repos.txt'

    # time_stamps = [f'{year}-{month:02d}' for year in range(2017, 2024) for month in range(1, 13)]
    time_stamps = [time_stamp]

    # pre_defined repos
    with open(repo_list, 'r') as f:
        repos = f.readlines()
    
    print(f"Total {len(repos)} repos")
    
    # Prepare URLs and local paths
    urls = [f'https://github.com/{repo.strip()}.git' for repo in repos]
    local_paths = [os.path.join(repo_path, repo.replace('/', '_')).strip() for repo in repos]

    # clone repos
    args = [(url, path, True, first_day_string) for url, path in zip(urls, local_paths)]
    for arg in args:
        clone_repo(*arg)
    # with multiprocessing.Pool(2) as pool:
    #     pool.starmap(clone_repo, args)
    
    code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.php', '.ts', '.jsx', '.tsx', '.css', '.sh', '.pl', '.bat'}
    
    combinations = list(itertools.product(time_stamps, local_paths))
    # combinations = sorted(combinations, key=lambda x: x[-1])
    flattened_args = [(time_stamp, local_path, save_dir) for time_stamp, local_path in combinations]

    print(f"Total {len(flattened_args)} combinations")
    with multiprocessing.Pool(2) as pool:
        ALL_PROCESSED = pool.starmap(main, flattened_args)
    
    print(f"Total {len(ALL_PROCESSED)} processed")
