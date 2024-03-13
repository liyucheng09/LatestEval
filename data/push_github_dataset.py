from glob import glob
import datasets
import os
import json

if __name__ == '__main__':
    hf_token = os.environ['HF_TOKEN']
    all_months = [f'{year}-{month:02}' for year in range(2017, 2024) for month in range(1, 13)]
    all_months += [f'2024-{month:02}' for month in range(1,3)]
    
    # try:
    #     exists_config = datasets.get_dataset_config_names('RealTimeData/code_alltime')
    # except datasets.exceptions.DatasetNotFoundError:
    #     exists_config = []
    #     pass

    for month in all_months:
        # if month in exists_config:
        #     continue
        code_paths = glob(f'/vol/research/lyc/github_dataset/{month}/*/*.json')
        all_codes = []
        for code in code_paths:
            with open(code, 'r') as f:
                all_codes.append(json.load(f))
        ds = datasets.Dataset.from_list(all_codes)
        print('='*20)
        print(f'Finished {month}')
        print(ds)
        ds.push_to_hub(f'RealTimeData/code_alltime', config_name = month, token=hf_token)
        print(f'Pushed {month} to hub')
