import datasets
import json
from glob import glob

if __name__ == '__main__':

    files = glob('/vol/research/lyc/math/*.json')
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        
        time_stamp = file.split('/')[-1].split('.')[0]
        if time_stamp not in ['2024-01', '2024-02']:
            continue

        all_instances = []
        for qa in data.values():
            instance = {}
            instance['question'] = qa['title']
            instance['question_id'] = qa['question_id']
            instance['score'] = qa['score']
            instance['link'] = qa['link']
            instance['body'] = qa['body']
            if 'answers' not in qa:
                continue
            instance['answers'] = [{'text': a['body'], 'score': a['score'], 'answer_id': a['answer_id']} for a in qa['answers']]

            verbolised = f"Question: {instance['question']}\n"
            for ans_index, ans in enumerate(instance['answers']):
                verbolised += f"Answer {ans_index + 1}: {ans['text']}\n"
            instance['verbolised'] = verbolised

            all_instances.append(instance)
        
        dataset = datasets.Dataset.from_list(all_instances)
        print(dataset)

        dataset.push_to_hub('RealTimeData/math_alltime', time_stamp)
        print(f"Pushed {time_stamp} to hub")