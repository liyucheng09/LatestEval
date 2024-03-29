import requests
import time
from datetime import datetime
import sys
import os
import json
import datasets

Overflow_API_KEY = os.environ['Overflow_API_KEY']

def fetch_answers_for_question(site, question_id):
    """Fetch answers for a given question ID."""
    url = f"https://api.stackexchange.com/2.3/questions/{question_id}/answers"
    params = {
        'order': 'desc',
        'sort': 'votes',
        'site': site,
        'filter': 'withbody',  # Include answer body
        'key': Overflow_API_KEY,
    }
    response = requests.get(url, params=params)
    data = response.json()
    # if data['quota_remaining'] < 10:
    #     print(f"Warning: API quota remaining is {data['quota_remaining']} for site {site} and question {question_id}.")
    #     time.sleep(10)
    print(f"Quota remaining: {data['quota_remaining']}")
    return data.get('items', [])

def batched_answer_fetch(site, questions):
    """Fetch answers for given question IDs in batches."""
    all_answers = []
    # API might have a limit on the number of IDs per request; adjust batch_size if needed
    answer_count = 0
    batch_ids = []
    question_and_answers = {}
    for question in questions:
        if question['answer_count'] < 2:
            continue
        question['site'] = site
        question_and_answers[question['question_id']] = question
        answer_count += question['answer_count']
        if answer_count < 100:
            batch_ids.append(question['question_id'])
            continue
        else:
            ids_string = ';'.join(map(str, batch_ids))
            url = f"https://api.stackexchange.com/2.3/questions/{ids_string}/answers"
            params = {
                'order': 'desc',
                'sort': 'votes',
                'site': site,
                'filter': 'withbody',  # Include answer body
                'key': Overflow_API_KEY, # Replace YOUR_API_KEY with your actual Stack Exchange API key
                'pagesize': 100,  # Adjust as per your needs, max is typically 100
            }
            response = requests.get(url, params=params)
            data = response.json()
            all_answers.extend(data.get('items', []))
            for ans in all_answers:
                question_id = ans['question_id']
                assert question_id in question_and_answers
                question_and_answers[question_id].setdefault('answers', []).append(ans)
            
            print(f"Quota remaining: {data['quota_remaining']}")
            batch_ids = [question['question_id']]
            answer_count = 0
                
    return question_and_answers

def fetch_questions_within_period(site, time_stamp, max_items=100):
    questions = []
    page = 1
    has_more = True
    year = int(time_stamp.split('-')[0])
    month = int(time_stamp.split('-')[1])
    start_date = datetime(year, month, 1)
    end_date = datetime(year, month, 28)

    fromdate = int(start_date.timestamp())
    todate = int(end_date.timestamp())
    
    while has_more and len(questions) < max_items:
        url = "https://api.stackexchange.com/2.3/questions"
        params = {
            'site': site,
            'sort': 'votes',
            'fromdate': fromdate,
            'todate': todate,
            'pagesize': 100,  # Adjust as per your needs, max is typically 100
            'page': page,
            'filter': 'withbody',  # Include question body
            'key': Overflow_API_KEY,
        }
        response = requests.get(url, params=params)
        data = response.json()
        questions.extend(data.get('items', []))
        has_more = data.get('has_more', False)
        page += 1
        time.sleep(0.05)
    
    return questions

if __name__ == '__main__':
    hf_token = os.environ['HF_TOKEN']

    today = datetime.today()
    year = today.year
    month = today.month
    time_stamp = f'{year}-{month:02d}'

    save_path = './math'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Example usage
    sites = ["math", "mathoverflow"]
    all_results = {}
    for site in sites:
        questions = fetch_questions_within_period(site, time_stamp, max_items=1800)
        print(f"Found {len(questions)} questions on {site} for {time_stamp}")

        question_and_answers = batched_answer_fetch(site, questions)
        all_results.update(question_and_answers)
    
    # Save the data
    file_path = os.path.join(save_path, f'{time_stamp}.json')
    with open(file_path, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(all_results)} questions and answers for {time_stamp} to {file_path}")

    # load the data
    with open(file_path) as f:
        data = json.load(f)

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

    ds = datasets.Dataset.from_list(all_instances)
    ds.push_to_hub('RealTimeData/math_alltime', time_stamp, token=hf_token)