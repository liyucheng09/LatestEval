import requests
from bs4 import BeautifulSoup
import praw
import sys
from typing import List
import json
import os
import time
import datetime
import traceback

class Forum:
    def __init__(self, task_name, start_url, wait_time):
        self.task_name = task_name
        self.url = start_url
        self.wait_time = wait_time

        self.session = requests.Session()
        self.setup_session()
    
        self.posts = None
    
    def setup_session(self):
        """
            _summary_: Setup session
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        self.session.headers.update(headers)
    
    def get_forum_pages(self):
        """
            _summary_: Get all pages links of the forum
        """
        return NotImplementedError()
    
    def get_forum_content(self, page_url):
        """
            _summary_: Get all content of the forum
            should have:
                - title
                - main content
                - comments
                - votes of comments
        """
        return NotImplementedError()
    
    def obtain_content(self):
        """
            _summary_: Obtain content from each post
        """
        list_of_content = []
        for post in self.posts:
            list_of_content.append(self.get_forum_content(post))
        
        self.content = list_of_content
    
    def save_content(self):
        """
            _summary_: Save list_of_content to a file
        """
        return NotImplementedError()
    
    @classmethod
    def filter_func(cls, tag, prefix):
        if tag.has_attr('class'):
            class_str = ' '.join(tag['class'])
            # return class_str.startswith('node node--id')
            return class_str.startswith(prefix)
        return False

class MentalHealth(Forum):
    def __init__(self, task_name, start_url, wait_time):
        super().__init__(task_name, start_url, wait_time)

    def get_forum_pages(self):
        # get all sub forums

        sub_forums = []
        response = self.session.get(self.url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all(lambda tag: self.filter_func(tag, 'node node--id')):
            sub_forums.append(link.find('a')['href'])
        
        # get all posts from each forum
        posts = []
        for sub_forum in sub_forums:
            response = self.session.get(sub_forum)
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all(lambda tag: self.filter_func(tag, 'structItem structItem--thread')):
                posts.append(link['href'])
        
        self.posts = posts

    def get_forum_content(self, page_url):
        response = self.session.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('h1', {'class': 'p-title-value'}).text
        list_of_content = []
        for article in soup.find_all(lambda tag: self.filter_func(tag, 'message message--post ')):
            author = article['data-author']
            content = article.find('div', {'class': 'bbWrapper'})
            if content:
                mentioned = content.find_all('a', {'class': 'username'})
                content = content.text
            else:
                continue
            
            footer = article.find('ul', {'class': 'sv-rating-bar__ratings'})
            if footer:
                ratings = footer.find_all('li', {'class': 'sv-rating sv-rating--empty-list'})
                rating_sum = sum([ int(rate.text) for rate in rating])
            else:
                rating_sum = 0
            
            list_of_content.append({
                'author': author,
                'content': content,
                'mentioned': mentioned,
                'rating': rating_sum
            })

        next_page = soup.find('a', {'class': 'pageNav-jump pageNav-jump--next'})
        if next_page:
            next_page = next_page['href']
            list_of_content += self.get_forum_content(next_page)

        return list_of_content      


class Reddit:
    def __init__(self, subreddits, time_filter, num_posts, save_path, time_limit = None):
        self.subreddits = subreddits
        self.time_filter = time_filter
        self.num_posts = num_posts
        self.save_path = save_path
        self.time_limit = time_limit

        self.reddit = praw.Reddit('DataCollector')
        self.posts = self.get_reddit_posts()
        # self.dump_posts()

    def created_after_time_limit(self, created_utc):
        if self.time_limit is None:
            return True
        dt_object = datetime.datetime.fromtimestamp(created_utc)
        return dt_object >= self.time_limit

    def get_reddit_posts(self):
        # all_posts = {}
        for subreddit in self.subreddits:
            subreddit_ = subreddit
            subreddit = self.reddit.subreddit(subreddit)
            list_of_posts = []
            for post in subreddit.top(time_filter = self.time_filter, limit=self.num_posts):
                created_time = post.created_utc
                if not self.created_after_time_limit(created_time): continue
                created_time_str = datetime.datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S')
                title = post.title
                content = post.selftext

                for i in range(3):
                    try:
                        comments = self.deal_with_comments(post.comments.list())
                    except praw.exceptions.APIException as e:
                        traceback.print_exc()
                        time.sleep(10)
                    else:
                        break

                score = post.score
                the_post = {
                    'title': title,
                    'content': content,
                    'comments': comments,
                    'created_time': created_time_str,
                    'score': score,
                    'subreddit': subreddit_
                }
                list_of_posts.append(the_post)
            self.dump_posts(list_of_posts, subreddit_)
            # all_posts[subreddit_] = list_of_posts
        # return all_posts
    
    def deal_with_comments(self, comments, depth = 3):
        results = []
        if depth < 0: return results
        depth -= 1
        for comment in comments:
            if isinstance(comment, praw.models.MoreComments): continue
            author = comment.author
            content = comment.body
            score = comment.score
            created_time = comment.created_utc
            created_time_str = datetime.datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S')
            replies = comment.replies
            if len(replies):
                replies = self.deal_with_comments(replies, depth=depth)
            else: replies = []
            the_comment = {
                'author': author.name if author is not None else '',
                'content': content,
                'score': score,
                'created_time': created_time_str,
                'replies': replies
            }
            results.append(the_comment)
        return results
    
    def dump_posts(self, list_of_posts, subreddit = None):
        path = os.path.join(self.save_path, f"{subreddit if subreddit is not None else 'all'}.json")
        with open(path, 'w') as f:
            json.dump(list_of_posts, f)

if __name__ == '__main__':
    # should define the XDG_CONFIG_HOME to the config file
    cwd, = sys.argv[1:]
    data_collectors = Reddit(['investing', 'wallstreetbets', 'CryptoCurrency', 'politics', 'healthcare'], 'month', 100, cwd, time_limit=datetime.datetime(2023, 7, 1))