from glob import glob
import json

if __name__ == '__main__':
    # /vol/research/lyc/bbc/2023/0/articles.1 indicates day 1, month 0, year 2023
    docs = glob('/vol/research/lyc/bbc/2023/*/articles.*')

    # now group by month
    months = {}
    for doc in docs:
        month = doc.split('/')[-2]
        month = int(month) + 1
        if month not in months:
            months[month] = []

        with open(doc, 'r') as f:
            articles = json.load(f)['articles']
            months[month].extend(articles)
    
    # now save
    # each month should save as a json dict
    # target path /vol/research/lyc/bbc/bbc_alltime/articles/2023-{month}.json
    for month in months:
        articles = months[month]
        # now turn list of dicts to dict of lists
        articles = { key: [article[key] for article in articles] for key in articles[0] }
        with open(f'/vol/research/lyc/bbc/bbc_alltime/articles/2023-{month}.json', 'w') as f:
            json.dump(articles, f, ensure_ascii=False)
        print(f'Finished month {month}')