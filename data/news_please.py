from newsplease import NewsPlease

# Set up news-please crawler
crawler = NewsPlease()

# List of seed news sites to crawl
sites = ['nytimes.com', 'washingtonpost.com', 'theguardian.com'] 

# Set date range for articles
start_date = '2023-01-01'
end_date = '2023-12-31'

# Collect articles
articles = []
for site in sites:
    articles.extend(crawler.get_news(site, count=5000, from_date=start_date, to_date=end_date))

# Post-process articles
processed_articles = []
for article in articles:
    # Filter out unwanted domains
    if article.source_domain not in sites:
        continue
    
    # Extract text and metadata
    text = article.maintext
    title = article.title
    authors = article.authors
    date = article.date_publish
    
    # Build structured example
    structured_article = {
        'text': text, 
        'title': title,
        'authors': authors,
        'date': date
    }
    
    processed_articles.append(structured_article)

# Split into train and test sets
train_articles = [] 
test_articles = []

for article in processed_articles:
   if article['date'] < '2020-10-01':
       train_articles.append(article)
   else:
       test_articles.append(article)
       
# Preprocess train set for LM training...

# Evaluate on test set...