from glob import glob
import datasets
import os
import json

if __name__ == '__main__':
    hf_token = os.environ['HF_TOKEN']
    all_months = [f'{year}-{month:02}' for year in range(2017, 2024) for month in range(1, 13)]

    # try:
    #     exists_config = datasets.get_dataset_config_names('RealTimeData/arxiv_alltime')
    # except datasets.exceptions.DatasetNotFoundError:
    #     exists_config = []
    #     pass

    # all months before 2021-02 (included) are already pushed, so remove these months from all_months
    all_months = all_months[all_months.index('2021-03'):]

    for month in all_months:
        # if month in exists_config:
        #     continue
        paper_paths = glob(f'/vol/research/lyc/arxiv_alltime/{month}/*.json')
        all_papers = []
        for paper in paper_paths:
            with open(paper, 'r') as f:
                all_papers.append(json.load(f))
        ds = datasets.Dataset.from_list(all_papers)
        print('='*20)
        print(f'Finished {month}')
        print(ds)
        ds.push_to_hub(f'RealTimeData/arxiv_alltime', config_name = month, token=hf_token)
        print(f'Pushed {month} to hub')
