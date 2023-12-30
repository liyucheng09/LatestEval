from glob import glob
import json

if __name__ == '__main__':
    # /vol/research/lyc/bbc/2023/0/articles.1 indicates day 1, month 0, year 2023
    docs = glob('/vol/research/lyc/bbc/*/*/articles.*')

    # now group by month
    times = {}
    for doc in docs:
        year = doc.split('/')[-3]
        month = doc.split('/')[-2]
        month = int(month)%12 + 1
        time = f'{year}-{month}'
        if time not in times:
            times[time] = []

        with open(doc, 'r') as f:
            articles = json.load(f)['articles']
            times[time].extend(articles)
    
    # now save
    # each month should save as a json dict
    # target path /vol/research/lyc/bbc/bbc_alltime/articles/2023-{month}.json
    for time in times:
        articles = times[time]
        month = time.split('-')[1]
        year = time.split('-')[0]
        # now turn list of dicts to dict of lists
        articles = { key: [article[key] for article in articles] for key in articles[0] }
        with open(f'/vol/research/lyc/bbc/bbc_alltime/articles/{year}-{month}.json', 'w') as f:
            json.dump(articles, f, ensure_ascii=False)
        print(f'Finished {year} {month}')