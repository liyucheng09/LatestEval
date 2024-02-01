from difflib import SequenceMatcher
import datasets
import multiprocessing

def compare_texts(text1, text2):
    # Split the texts into words
    words1 = text1.split()
    words2 = text2.split()

    # Create a SequenceMatcher to compare the two word lists
    matcher = SequenceMatcher(None, words1, words2)

    # Calculate the similarity ratio
    similarity = matcher.ratio()

    # Calculate the difference ratio
    difference = 1 - similarity

    return difference

def main(month, first_month_articles):
    ds = datasets.load_dataset('RealTimeData/wikitext_alltime', month, split='train')
    # compare to first month
    diffs = []
    for article in ds:
        title = article['title']
        text = article['text']
        if title not in first_month_articles:
            print(f"Article {title} not found in first month")
            continue
        first_month_text = first_month_articles[title]
        difference = compare_texts(text, first_month_text)
        diffs.append(difference)
    
    avg_diff = sum(diffs) / len(diffs)
    print(f"Finished {month}, average difference: {avg_diff}")
    return (month, avg_diff)

if __name__ == '__main__':
    
    months = [f'{year}-{month:02d}' for year in range(2017, 2024) for month in range(1, 13) if not (year == 2023 and month == 12)]
    first_month = datasets.load_dataset('RealTimeData/wikitext_alltime', months[0], split='train')
    first_month_articles = {title: article for title, article in zip(first_month['title'], first_month['text'])}
    diffs = {}

    months = months[1:]
    # main got two arguments, month and first_month_articles
    # pool size 4
    with multiprocessing.Pool(8) as pool:
        for month, diff in pool.starmap(main, [(month, first_month_articles) for month in months]):
            diffs[month] = diff

    print(diffs)