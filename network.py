"""
This module contains fetchers for the Avatar domain.
"""
from typing import Tuple, Set, Any, Optional
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pickle
import logging

logger = logging.getLogger(__name__)


MAX_THREADS = 100

def fetch_url(url: str) -> requests.Response:
    _response = requests.get(url)
    return _response

def fetch_urls_out_of_response(_response: requests.Response) -> set:
    soup = BeautifulSoup(_response.text, 'html.parser')
    # lets set soup to only be the div with class "page_main"
    soup = soup.find('main', class_='page__main')
    references_span = soup.find('span', id='References')
    if references_span:
        # this tag father is the h2 tag above the table of references so we can get the table of references
        references_table = references_span.find_next('table')
        if references_table:
            references_table.decompose()

    _urls = set()
    count = 0
    for link in soup.find_all('a'):
        count += 1
        if link.get('href') is None:
            continue
        endpoint = link.get('href')
        if '#' in endpoint:
            endpoint = endpoint.split('#')[0]
            if not endpoint:
                continue
        # Lets make sure the endpoint is not a language link by checking if there is a word between com and /wiki
        if '.com/' in endpoint:
            check: str = endpoint.split('.com/')[1]
            if not check.startswith('wiki'):
                continue
        if 'Category' in endpoint or '?' in endpoint\
                or 'Talk:' in endpoint or 'Avatar_Wiki:' in endpoint or 'File' in endpoint\
                or 'Special' in endpoint or 'Help:' in endpoint or 'Template' in endpoint\
                or 'User:' in endpoint or 'User_blog:' in endpoint or 'Thread:' in endpoint\
                or 'military-history.fandom.com' in endpoint:
            continue
        _urls.add(endpoint)
    return _urls

def set_urls_to_full(_urls: set) -> set:
    _new_urls = set()
    for _url in _urls:
        if 'https://avatar.fandom.com' not in _url:
            _url = 'https://avatar.fandom.com' + _url
        _new_urls.add(_url)
    return _new_urls

def fetch_current_all_pages_urls(response: requests.Response) -> tuple[set[Any], Optional[Any]]:
    """
    This function responsible for fetching all the pages from the current page of all data.
    It will return the urls and the next page as well
    :return:
    """
    soup = BeautifulSoup(response.text, 'html.parser')
    main = soup.find('div', class_='mw-allpages-body')
    urls = set()
    for link in main.find_all('a'):
        if link.get('href') is None:
            continue
        endpoint = link.get('href')
        if 'Netflix' in endpoint or 'Category' in endpoint or '?' in endpoint\
                or 'Talk:' in endpoint or 'Avatar_Wiki:' in endpoint or 'File' in endpoint\
                or 'Special' in endpoint or 'Help:' in endpoint or 'Template' in endpoint\
                or 'User:' in endpoint or 'User_blog:' in endpoint or 'Thread:' in endpoint\
                or 'military-history.fandom.com' in endpoint:
            continue
        urls.add(endpoint)
    next_page_div = soup.find('div', class_='mw-allpages-nav')
    next_page = None
    for link in next_page_div.find_all('a'):
        next_page = link.get('href')
    return urls, next_page

def fetch_all_pages():
    total_urls = set()
    next_pages_set = set()
    next_page = 'https://avatar.fandom.com/wiki/Special:AllPages'
    counter = 0
    while next_page is not None:
        _response = fetch_url(next_page)
        _new_urls, next_page = fetch_current_all_pages_urls(_response)

        _urls = set_urls_to_full(_new_urls)
        next_page = 'https://avatar.fandom.com' + next_page if 'https://avatar.fandom.com'\
                                                               not in next_page else next_page
        total_urls = total_urls.union(_urls)
        counter += len(_urls)
        if next_page in next_pages_set:
            break
        next_pages_set.add(next_page)
        print(f'Counter: {counter}, next_page: {next_page}')


    return total_urls

def process_url(url):
    response = fetch_url(url)
    if url != response.url:
        print(f'{url} is redirected to {response.url}')
        return {'Redirected', response.url}
    urls = fetch_urls_out_of_response(response)
    urls = set_urls_to_full(urls)
    open_paragraphs = fetch_open_paragraph(response)
    return urls, open_paragraphs

def page_to_pages_mapper(_total_urls: set):
    _mapper = defaultdict(set)
    _mapper_for_open_paragraphs = defaultdict(str)
    length = len(_total_urls)
    counter = 0
    # lets make this concurrent using ThreadPoolExecutor
    # Lets use 10 threads
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(process_url, url): url for url in _total_urls}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                if isinstance(result, tuple):
                    urls, open_paragraphs = result
                    _mapper[url] = urls
                    _mapper_for_open_paragraphs[url] = open_paragraphs
                else:
                    # Dealing with the redirected urls
                    _mapper[url] = result
            except Exception as exc:
                print(f'{url} generated an exception: {exc}')
            finally:
                counter += 1
                print(f'Counter: {counter}/{length}')

    return _mapper, _mapper_for_open_paragraphs

def get_mapper(from_web: bool = False):
    if from_web:
        total_urls = fetch_all_pages()
        mapper, paragraphs_mapper = page_to_pages_mapper(total_urls)
        # testing code
        with open('_mapper.pkl', 'wb') as f:
            pickle.dump(mapper, f)
        with open('_paragraphs_mapper.pkl', 'wb') as f:
            pickle.dump(paragraphs_mapper, f)
    else:
        with open('_mapper.pkl', 'rb') as f:
            mapper = pickle.load(f)
        with open('_paragraphs_mapper.pkl', 'rb') as f:
            paragraphs_mapper = pickle.load(f)
    return mapper, paragraphs_mapper

def fetch_open_paragraph(response):
    soup = BeautifulSoup(response.text, 'html.parser')
    main = soup.find('main', class_='page__main')
    paragraphs = main.find_all('p')
    if paragraphs and isinstance(paragraphs, list):
        first_paragraph = paragraphs[0]
        aside = first_paragraph.find('aside')
        if aside:
            aside.decompose()
        text = first_paragraph.text
        for i in range(1, len(paragraphs)):
            text += paragraphs[i].text
        return text
    return ''


# response = fetch_url('https://avatar.fandom.com/wiki/Hundred_Year_War')
# urls = set_urls_to_full(fetch_urls_out_of_response(response))
# print(urls)
# lets check that url has inly one time the symbol :