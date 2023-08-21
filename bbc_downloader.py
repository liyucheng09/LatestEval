import weakref
import requests

from configobj import ConfigObj

class Configuration:

    def __init__(self):
        self.__properties = dict()
        properties = self._init_properties()
        for property_, value, transform_fn in properties:
            if transform_fn is not None:
                value = transform_fn(value)
            setattr(self, property_, value)
            self.__properties[property_] = {
                'default-value': value,
                'transform_fn': transform_fn
            }

    def _init_properties(self):
        # [[name, default-value, transform_fn]]
        return []

    # TODO: hierachical config
    def load(self, path):
        config = ConfigObj(path, encoding='UTF-8')
        for property_, value in config.items():
            transform_fn = self.__properties[property_]['transform_fn']
            if transform_fn is not None:
                value = transform_fn(value)
            setattr(self, property_, value)

from dateutil.relativedelta import relativedelta
# from datetime import datetime, date
import datetime

class DatasetConfiguration(Configuration):

    def _format_date(self, date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d')

    def _calculate_step(self, step):
        step = int(step)
        if self.step_unit == 'day':
            return relativedelta(days=step)
        elif self.step_unit == 'month':
            return relativedelta(months=step)
        else:
            return relativedelta(years=step)

    def _init_properties(self):
        return [
            ['name', '', str],
            ['base_api_url', 'http://dracos.co.uk/made/bbc-news-archive/{year}/{month:0>2}/{day:0>2}/', str],
            ['start_date', '2016-01-01', self._format_date],
            ['end_date', '2017-01-01', self._format_date],
            ['step_unit', 'day', str],
            ['step', 1, self._calculate_step],
            ['path', './dataset/bbc/', str],
            ['sleep', 1, float]
        ]

class NetWorkConfiguration(Configuration):

    HTTP_TIMEOUT = 30
    STRICT = True
    USER_AGENT = 'Mozilla'

    def _init_properties(self):
        return [
            ['browser_user_agent', 'Mozilla', str],
            ['http_timeout', 30, int],
            ['strict', True, lambda v: str(v) == 'True']
        ]

class NetworkError(RuntimeError):

    def __init__(self, status_code, reason):
        self.reason = reason
        self.status_code = status_code

class NetworkFetcher(object):

    def __init__(self):
        self.config = NetWorkConfiguration()
        # self.config.load('./settings/network.cfg')
        self.config.strict = False

        self._connection = requests.Session()
        self._connection.headers['User-agent'] = self.config.browser_user_agent
        self._finalizer = weakref.finalize(self, self.close)

        self._url = None
        self.response = None
        self.headers = None

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def get_url(self):
        return self._url

    def fetch(self, url):
        try:
            response = self._connection.get(url, timeout=self.config.http_timeout, headers=self.headers)
        except Exception:
            return None
        if response.ok:
            self._url = response.url
            text = response.content
        else:
            self._url = None
            text = None
            if self.config.strict:
                raise NetworkError(response.status_code, response.reason)

        return text

class DownloadLinkFetcher:

    RETRY = 5

    def __init__(self, config):
        self.base_api_url = config.base_api_url

        self.start_date = config.start_date
        self.current_date = config.start_date
        self.end_date = config.end_date
        self.step_unit = config.step_unit
        self.step = config.step

        self.html_fetcher = NetworkFetcher()

    def _format_link(self, link):
        print(link)
        hash_index = link.find('#')
        if hash_index != -1:
            link = link[:hash_index]
        if link[-1] == '/':
            link = link[:-1]
        return link

    def _link_filter(self, link, filters):
        if not link[-1].isdigit():
            return False
        for filter_ in filters:
            if link[filter_[1]:filter_[2]] == filter_[0]:
                return False
        return True

    def _html_to_links(self, html):
        return []

    def _next_api(self, base_url, current_date):
        return ''

    def next(self):
        if self.current_date >= self.end_date:
            return None, None
        api_url = self._next_api(self.base_api_url, self.current_date)
        date = self.current_date
        self.current_date += self.step
        return api_url, date

    def fetch(self, api_url):
        print('fetching download links...')
        html = self.html_fetcher.fetch(api_url)
        if html is None:
            for _ in range(0, self.RETRY):
                html = self.html_fetcher.fetch(api_url)
                if html is not None:
                    break
        if html is None or len(html) == 0:
            print('api', api_url, ' failed')
            return []
        links = self._html_to_links(html)
        return links

from bs4 import BeautifulSoup

class BBCLinkFetcher(DownloadLinkFetcher):

    BBC_FILTERS = [
        ['programmes', 21, 31],
        ['correspondents', 26, 40],
        ['iplayer', 21, 28],
        ['radio', 21, 26],
        ['live', 27, 31],
        ['m', 7, 8],
        ['video_and_audio', 26, 41]
    ]

    def _next_api(self, base_url, current_date):
        year = current_date.year
        month = current_date.month
        day = current_date.day
        api_url = base_url.format(year=year, month=month, day=day)
        return api_url

    def _html_to_links(self, html):
        soup = BeautifulSoup(html, 'lxml')

        links = list()
        # news links are the hrefs of a
        elements = soup.table.find_all('a')
        # elements = soup.table.find_all('a', class_='title-link')
        for element in elements:
            if not element['href']:
                continue
            link = self._format_link(element['href'])
            if self._link_filter(link, self.BBC_FILTERS):
                links.append(link)

        return list(set(links))


import sys
import os.path
import json
import time
from datetime import timedelta

class ArticleFetcher:

    RETRY = 5

    def __init__(self, config):
        self.config = config
        self.download_link_fetcher = None
        self.html_fetcher = NetworkFetcher()
        self.path = config.path

        self.total_date = 0

        self._mkdir(self.path,
                    config.start_date,
                    config.end_date,
                    config.step)

    def _mkdir(self, path, start_date, end_date, step):
        if os.path.isdir(path):
            # current_date = start_date
            # while current_date < end_date:
            #     current_date += step
            #     self.total_date += 1
            # return
            pass
        else:
            os.makedirs(path)
        current_date = start_date
        existed_years = dict()
        while current_date < end_date:
            year = current_date.year
            month = current_date.month
            day = current_date.day

            year_path = os.path.join(path, str(year))
            month_path = os.path.join(year_path, str(month))
            day_path = os.path.join(month_path, str(day))

            if year not in existed_years.keys():
                existed_years[year] = dict()
                if not os.path.isdir(year_path):
                    os.mkdir(year_path)

            if (step.months > 0) or (step.days > 0):
                year_content = existed_years[year]
                if month not in year_content.keys():
                    year_content[month] = True
                    if not os.path.isdir(month_path):
                        os.mkdir(month_path)

            if step.days > 0:
                if not os.path.isdir(day_path):
                    os.mkdir(day_path)
            current_date += step

            self.total_date += 1

    def _html_to_infomation(self, html, link, date):
        return {}

    def _extract_information(self, link, date):
        html = self.html_fetcher.fetch(link)
        if html is None:
            for _ in range(0, self.RETRY):
                html = self.html_fetcher.fetch(link)
                if html is not None:
                    break
        if html is None:
            print('article ', link, 'failed')
            return None
        return self._html_to_infomation(html, link, date)

    def _get_storage_path(self, path, date):
        return os.path.join(path, str(date.year), str(date.month), str(date.day))

    def _lazy_storage(self, storage_path, links, date, current_date):
        total_links = len(links)
        current_link = 1

        titles_path = os.path.join(storage_path, f'titles.{current_date}')
        with open(titles_path, mode='w', encoding='utf-8') as titles_file:
            articles = list()
            titles = list()
            for link in links:
                print('>>> {c} in {t} articles\r'.format(c=current_link, t=total_links), end='')
                current_link += 1

                article = self._extract_information(link, date)
                if article is not None:
                    titles.append(article['title'] + '\n')
                    articles.append(article)

            articles_path = os.path.join(storage_path, f'articles.{current_date}')
            with open(articles_path, mode='w', encoding='utf-8') as articles_file:
                json.dump({
                    'expected_number': len(links),
                    'number': len(articles),
                    'articles': articles
                }, articles_file, indent=4)
            titles_file.writelines(titles)

    def _non_lazy_storage(self, storage_path, links, date):
        total_links = len(links)
        current_link = 1

        titles_path = os.path.join(storage_path, 'titles')
        with open(titles_path, mode='w', encoding='utf-8') as titles_file:
            for article_index, link in enumerate(links):
                print('{c} in {t} articles\r'.format(c=current_link, t=total_links), end='')
                current_link += 1

                article = self._extract_information(link, date)
                if article is not None:
                    titles_file.write(article['title'] + '\n')

                    article_path = os.path.join(storage_path, str(article_index))
                    with open(article_path, mode='w', encoding='utf-8') as article_file:
                        json.dump(article, article_file, indent=4)

    def fetch(self, lazy_storage=True):
        current_date = 1
        while True:
            api_url, date = self.download_link_fetcher.next()
            if api_url is None:
                break
            print(date.strftime('%Y-%m-%d'),
                  '{c} in {t} dates                  '.format(c=current_date, t=self.total_date))

            # storage_path = self._get_storage_path(self.path, date)
            storage_path = self.path
            links = self.download_link_fetcher.fetch(api_url)
            if lazy_storage:
                self._lazy_storage(storage_path, links, date, current_date)
            else:
                self._non_lazy_storage(storage_path, links, date)

            time.sleep(self.config.sleep)

            print(date.strftime('%Y-%m-%d'),
                  'date {c} finished                 '.format(c=current_date))
            current_date += 1

import json

from bs4 import BeautifulSoup
from goose3 import Goose
from goose3.extractors.content import ContentExtractor

eps = 1e-6
f1 = ContentExtractor.calculate_best_node
f2 = ContentExtractor.post_cleanup


def post_cleanup(ce_inst):
    """\
    remove any divs that looks like non-content,
    clusters of links, or paras with no gusto
    """
    parse_tags = ['p']
    if ce_inst.config.parse_lists:
        parse_tags.extend(['ul', 'ol'])
    if ce_inst.config.parse_headers:
        parse_tags.extend(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    target_node = ce_inst.article.top_node
    node = ce_inst.add_siblings(target_node)
    for elm in ce_inst.parser.getChildren(node):
        e_tag = ce_inst.parser.getTag(elm)
        if e_tag not in parse_tags:
            if ce_inst.is_highlink_density(elm) or ce_inst.is_table_and_no_para_exist(elm):
                ce_inst.parser.remove(elm)
    return node


def calculate_best_node(ce_inst, doc):
    top_node = None
    nodes_to_check = ce_inst.nodes_to_check(doc)

    starting_boost = float(1.0)
    cnt = 0
    i = 0
    parent_nodes = []
    nodes_with_text = []

    for node in nodes_to_check:
        text_node = ce_inst.parser.getText(node)
        word_stats = ce_inst.stopwords_class(language=ce_inst.get_language()).get_stopword_count(text_node)
        high_link_density = ce_inst.is_highlink_density(node)
        if word_stats.get_stopword_count() > 2 and not high_link_density:
            nodes_with_text.append(node)

    nodes_number = len(nodes_with_text)
    negative_scoring = 0
    bottom_negativescore_nodes = float(nodes_number) * 0.25

    for node in nodes_with_text:
        boost_score = float(0)
        # boost
        if ce_inst.is_boostable(node):
            if cnt >= 0:
                boost_score = float((1.0 / starting_boost) * 50)
                starting_boost += 1
        # nodes_number
        if nodes_number > 15:
            if (nodes_number - i) <= bottom_negativescore_nodes:
                booster = float(bottom_negativescore_nodes - (nodes_number - i))
                boost_score = float(-pow(booster, float(2)))
                negscore = abs(boost_score) + negative_scoring
                if negscore > 40:
                    boost_score = float(5)

        text_node = ce_inst.parser.getText(node)
        word_stats = ce_inst.stopwords_class(language=ce_inst.get_language()).get_stopword_count(text_node)
        upscore = int(word_stats.get_stopword_count() + boost_score)

        # parent node
        parent_node = ce_inst.parser.getParent(node)
        ce_inst.update_score(parent_node, upscore)
        ce_inst.update_node_count(parent_node, 1)

        if parent_node not in parent_nodes:
            parent_nodes.append(parent_node)

        # parentparent node
        parent_parent_node = ce_inst.parser.getParent(parent_node)
        if parent_parent_node is not None:
            ce_inst.update_node_count(parent_parent_node, 1)
            ce_inst.update_score(parent_parent_node, upscore - eps)
            if parent_parent_node not in parent_nodes:
                parent_nodes.append(parent_parent_node)

        # parentparentparent node
        parent_parent_parent_node = ce_inst.parser.getParent(parent_parent_node)
        if parent_parent_parent_node is not None:
            ce_inst.update_node_count(parent_parent_parent_node, 1)
            ce_inst.update_score(parent_parent_parent_node, upscore - 2 * eps)
            if parent_parent_parent_node not in parent_nodes:
                parent_nodes.append(parent_parent_parent_node)
        cnt += 1
        i += 1

    top_node_score = 0
    for itm in parent_nodes:
        score = ce_inst.get_score(itm)

        if score > top_node_score:
            top_node = itm
            top_node_score = score

        if top_node is None:
            top_node = itm

    return top_node


class BBCArticleFetcher(ArticleFetcher):

    def __init__(self, config):
        super(BBCArticleFetcher, self).__init__(config)
        self.download_link_fetcher = BBCLinkFetcher(config)

    def _extract_title(self, soup):
        if soup.title is not None:
            return soup.title.get_text()

    def _extract_published_date(self, date):
        return date.strftime('%Y-%m-%d')

    def _extract_authors(self, soup):
        authors_elements = soup.find_all('meta', property='article:author')
        if authors_elements is not None:
            return [authors_element['content'] for authors_element in authors_elements]

    def _extract_description(self, soup):
        description_element = soup.find('meta', property='og:description')
        if description_element is not None:
            return description_element['content']

    def _extract_section(self, soup):
        section_element = soup.find('meta', property='article:section')
        if section_element is not None:
            return section_element['content']

    def _extract_content(self, html):
        ContentExtractor.calculate_best_node = calculate_best_node
        ContentExtractor.post_cleanup = post_cleanup
        g = Goose({'enable_image_fetching': False})
        article = g.extract(raw_html=html)
        ContentExtractor.calculate_best_node = f1
        ContentExtractor.post_cleanup = f2
        return article.cleaned_text

    def _html_to_infomation(self, html, link, date):
        soup = BeautifulSoup(html, 'lxml')
        head = soup.head

        try:
            title = self._extract_title(head)
            published_date = self._extract_published_date(date)
            authors = self._extract_authors(head)
            description = self._extract_description(head)
            section = self._extract_section(head)
            content = self._extract_content(html)
        except Exception:
            return None

        return {
            'title': title,
            'published_date': published_date,
            'authors': authors,
            'description': description,
            'section': section,
            'content': content,
            'link': link
        }

if __name__ == '__main__':

    today = datetime.date.today()
    today_str = today.strftime('%Y-%m-%d')
    two_weeks_ago = today - datetime.timedelta(days=14)
    two_weeks_ago_str = two_weeks_ago.strftime('%Y-%m-%d')

    config = DatasetConfiguration()
    config.start_date = two_weeks_ago
    config.end_date = today
    config.path = 'dataset/bbc'

    bbc_article_fetcher = BBCArticleFetcher(config)
    bbc_article_fetcher.fetch()

    from glob import glob
    files = glob(f'dataset/bbc/articles.*')
    files.sort()

    import datasets
    import json
    import os

    hf_token = os.environ['HF_TOKEN']

    all_articles = []
    for file in files:
        with open(file) as f:
            articles = json.load(f)

        articles = articles['articles']
        for article in articles:
            article['authors'] = article['authors'][0] if article['authors'] else None
            all_articles.append(article)
    
    with open('all_articles.json', 'w') as f:
        json.dump(all_articles, f, indent=4, ensure_ascii=False)
    
    ds = datasets.Dataset.from_dict({key: [article[key] for article in all_articles] for key in all_articles[0].keys()})
    ds.save_to_disk('bbc')

    from huggingface_hub import create_branch, create_tag, RepoCard

    create_branch('RealTimeData/bbc_latest', repo_type='dataset', branch=today_str)
    ds.push_to_hub('RealTimeData/bbc_latest', token=hf_token, branch='main')
    ds.push_to_hub('RealTimeData/bbc_latest', token=hf_token, branch=today_str)

    text = f"""
# Latest BBC News

This dataset contains the latest BBC News articles submitted from {two_weeks_ago.isoformat()} to {today.isoformat()}.

The data collection is conducted on {today.isoformat()}.

Use the dataset via:
```
ds = datasets.load_dataset('RealTimeData/bbc_latest')
```

# Previsou versions

You could access a historical slice from different branches.

For example, you could find the 2023-08-20 version via:
```
ds = datasets.load_dataset('RealTimeData/bbc_latest', revision = '2023-08-20')
```
"""
    card = RepoCard(text)
    card.push_to_hub('RealTimeData/bbc_latest', repo_type='dataset', token=hf_token)