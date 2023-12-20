# Merge with RealTimeData/ and push to Huggingface Hub

from glob import glob
import datasets
import json
from huggingface_hub import RepoCard, create_branch, create_tag
from data_processor import ArxivEval, BBCNewsEval, GithubEval
import datetime

if __name__ == "__main__":
    # Load the dataset
    # for example, benchmarks/latest/qa_pairs_arxiv_2023-46.json

    today = datetime.date.today()

    RepoCardText = """
# LatestEval for {source}

This benchmark was created with at {year} week {week} with the latest data from {source}.

check more details at our [github page](https://github.com/liyucheng09/LatestEval)."""

    source2ds = {}
    latest_ds = []

    for file in glob('benchmarks/2023-51/*.json'):
        with open(file, 'r') as f:
            data = json.load(f)

        if 'arxiv' in file:
            source = 'arxiv'
            docs = ArxivEval('RealTimeData/arxiv_latest', num_docs='all').docs
        elif 'bbc' in file:
            source = 'bbc'
            docs = BBCNewsEval('RealTimeData/bbc_latest', num_docs='all').docs
        elif 'github' in file:
            source = 'github'
            docs = GithubEval('RealTimeData/github_latest', num_docs='all').docs

        source2ds[source] = data
        
        time_stamp = file.split('_')[-1].split('.')[0]
        year = time_stamp.split('-')[0]
        week = time_stamp.split('-')[1]

        test_samples = []
        for doc in data:
            doc_id = doc['id'][len(source)+1:]
            sents = None
            for d in docs:
                if d.entry_id == doc_id:
                    sents = d.original_sentences
            assert sents is not None, f'{doc_id} not found in {source} data'

            if isinstance(doc['response'], str):
                try:
                    doc['response'] = eval(doc['response'])
                except:
                    print(doc['response'])
                    continue
            
            for example in doc['response']:
                sent_index = example['sentence_index']
                passage = ''
                for sent_i, sent in enumerate(sents):
                    if sent_i == sent_index:
                        passage += example['place_holder'] + ' '
                    else:
                        passage += sent + ' '
                test_samples.append({
                    'source': source,
                    'doc_id': doc_id,
                    'passage': passage,
                    'query': example['query'],
                    'answer': example['key_information'],
                    'query_category': example['answer_type'],
                    'sent_index': sent_index
                })

        latest_ds.extend(test_samples)

        # dataset = datasets.Dataset.from_list(test_samples)
        # dataset.push_to_hub(f'LatestEval/{source}-latest', branch='main')
        # dataset.push_to_hub(f'LatestEval/{source}-{year}-week{week}')

        # card = RepoCard(RepoCardText.format(source=source, year=year, week=week))
        # card.push_to_hub(f'LatestEval/{source}-latest', repo_type='dataset')
        # card.push_to_hub(f'LatestEval/{source}-{year}-week{week}', repo_type='dataset')

    # all three sources together
    # flatten the data and add source column

    dataset = datasets.Dataset.from_list(latest_ds)
    dataset.push_to_hub(f'LatestEval/full-latest', branch='main')
    dataset.push_to_hub(f'LatestEval/full-{year}-week{week}')

    card = RepoCard(RepoCardText.format(source='all', year=year, week=week))
    card.push_to_hub(f'LatestEval/full-latest', repo_type='dataset')
    card.push_to_hub(f'LatestEval/full-{year}-week{week}', repo_type='dataset')
