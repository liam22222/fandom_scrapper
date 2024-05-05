import re
from collections import defaultdict
import copy

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

from network import fetch_url, fetch_open_paragraph, get_mapper


def get_redirects_mapper(_mapper: dict):
    """
    This function is responsible for getting all the redirects from the mapper
    and clean the mapper from the redirects
    :param _mapper:
    :return:
    """
    _redirects = defaultdict(set)
    _mapper_copy = copy.deepcopy(_mapper)
    for url, urls in _mapper_copy.items():
        if len(urls) == 2 and 'Redirected' in urls:
            # lets get the other value in the set which is the redirected url
            urls = urls.difference({'Redirected'})
            urls = urls.pop()

            # We can get rid of the url that was redirected from mapper
            del _mapper[url]

            # Before we add it, lets check that the redirected url is in the mapper
            if urls not in _mapper:
                continue
            _redirects[url] = urls

    return _redirects

def create_reverse_mapper(_mapper: dict, _redirects: dict):
    """
    This function is responsible for creating a reverse mapper which means that for every key in this mapper
    it will have a set of values that are pointing to it
    :param _mapper: The mapper that we have of each url and what he is pointing to
    :param _redirects: The redirects that we have
    :return:
    """
    _reverse_mapper = defaultdict(set)
    for url, urls in _mapper.items():
        for _url in urls:
            if _url in _redirects:
                _url = _redirects[_url]
            _reverse_mapper[_url].add(url)
    return _reverse_mapper

def clean_paragraphs_text(text: str):
    if not text:
        return ''
    text = text.split('\n')
    text = [line.strip() for line in text if line.strip()]
    text = [line for line in text if len(line.split()) > 10]

    text = '\n'.join(text)
    text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers

    # Handle possessives: change "Aang's" to "Aang"
    text = re.sub(r"'s\b", "", text)

    # Remove other non-word characters, preserve spaces and apostrophes
    text = re.sub(r'[^\w\s]', '', text)

    text = text.lower()
    return text

def tokenize_text(text: str):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return lemmatized_tokens

def training_model(sentences: list, percentage_wanted: int = 10):
    print('Training model')
    counter = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            counter[word] += 1
    sort_counter = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    # We only care about the 10% most common words, so lets find the 10% most appearing word number of appearances
    # and use it in the min_count parameter
    total_words = len(sort_counter)
    division_factor = 100 // percentage_wanted
    ten_percent = total_words // division_factor
    min_count = list(sort_counter.values())[ten_percent]
    model = Word2Vec(sentences, vector_size=1000, window=500, min_count=min_count, workers=16)
    model.save('word2vec.model')
    print('Model trained and saved')
    return model

def get_n_most_similar_words(model, word: str, n: int):
    return model.wv.most_similar(word, topn=n)

def getting_text_to_model():
    counter = 0
    mapper, paragraph_mapper = get_mapper()
    total_len = len(paragraph_mapper)
    clean_text_mapper = {}
    for url, text in paragraph_mapper.items():
        new_text = clean_paragraphs_text(text)
        process_text = tokenize_text(new_text)
        counter += 1
        print(f'{counter}/{total_len} - {url} - {process_text}')
        clean_text_mapper[url] = process_text
    return clean_text_mapper

def get_opposite_phrase_score(phrase: str, vocab: dict):
    """
    This function is responsible for getting the opposite phrase of the phrase that we have ratio
    The idea is that phrases that are not truely a phrase will have a low ratio cause they can be found opposite
    for example, "sokka_katara" will have a low ratio cause "katara_sokka" will also appear.
    Yet, "fire_nation" will have a high ratio cause "nation_fire" will not appear in the text
    if the opposite phrase is not found, we will return None for it
    :param vocab:
    :param phrase:
    :return:
    """
    words = phrase.split('_')
    opposite_phrase = '_'.join(reversed(words))
    if opposite_phrase in vocab:
        return vocab[phrase] / vocab[opposite_phrase]
    return None

def get_phrase_ratio(p: str, vocab: dict):
    """
    This function is responsible for getting for the phrase ratio to the words
    That creates it
    For example, for the phrase "air_nomad" we will count the number of times that "air" and "nomad" appear in the text
    and the number of times that "air_nomad" appears in the text and we will return the ratio between them
    :param p: The phrase that we want to get the ratio for
    :param vocab: The vocab that we have
    :return: The ratio between the phrase and the words that create it
    """
    words = p.split('_')
    phrase_count = vocab[p] * len(words)
    words_count = sum([vocab[word] for word in words])
    return phrase_count / words_count

def sort_and_normalize_dict(d: dict):
    """
    This function takes dict where value is countable and returns a dict where the values are normalized and sorted
    :param d:
    :return:
    """
    max_value = max(d.values())
    min_value = min(d.values())
    new_dict = {}
    for key, value in d.items():
        new_dict[key] = (value - min_value) / (max_value - min_value)
    return dict(sorted(new_dict.items(), key=lambda x: x[1], reverse=True))

def filter_phrases(phrases_dict: dict, vocab: dict):
    new_phrases_dict = copy.deepcopy(phrases_dict)
    phrase_ratio = dict()
    phrases_counter = dict()
    opposite_phrases = dict()
    for phrase, score in phrases_dict.items():
        if score > 35:
            continue
        ratio = get_phrase_ratio(phrase, vocab)
        phrase_ratio[phrase] = ratio
        phrases_counter[phrase] = vocab[phrase]
        opposite_phrases[phrase] = get_opposite_phrase_score(phrase, vocab)

    phrases_counter = sort_and_normalize_dict(phrases_counter)
    phrase_score = {phrase: phrases_counter[phrase] * phrase_ratio[phrase] for phrase in phrases_counter}
    phrase_score = sort_and_normalize_dict(phrase_score)
    max_opposite = max([value for value in opposite_phrases.values() if value is not None])
    for phrase, score in phrase_score.items():
        if opposite_phrases[phrase] is None:
            opposite_phrases[phrase] = max_opposite
    opposite_phrases = sort_and_normalize_dict(opposite_phrases)


    for phrase, score in phrase_score.items():
        if score < 0.1:
            del new_phrases_dict[phrase]
        elif opposite_phrases[phrase] < 0.15:
            del new_phrases_dict[phrase]
    return new_phrases_dict

def nlp_adjustment(sentences: list):
    nlp = spacy.load("en_core_web_sm")
    keep_tags = {'PROPN', 'NOUN', 'VERB', 'X'}
    saver = defaultdict(list)
    new_sentences = []
    size = len(sentences)
    for i, sentence in enumerate(sentences):
        if not isinstance(sentence, list):
            continue
        whole_sentence = ' '.join(sentence)
        doc = nlp(whole_sentence)
        new_sentence = []
        for token in doc:
            saver[token.pos_].append(token.text)
            if '_' in token.text:
                new_sentence.append(token.text)
                continue
            if token.pos_ in keep_tags:
                new_sentence.append(token.text)
        new_sentences.append(new_sentence)
        print(f'{i}/{size} - sentences done')
    return new_sentences



def getting_model(train: bool = False, number_of_words_per_phrase: int = 3, nlp_adjust: bool = False):
    if train:
        clean_text_mapper = getting_text_to_model()
        sentences = list(clean_text_mapper.values())
        if number_of_words_per_phrase != -1:
            phrases_found = set()
            for n in range(2, number_of_words_per_phrase + 1):
                phrases = Phrases(sentences, min_count=10, threshold=10)
                n_gram = Phraser(phrases)
                phrases_dict = copy.deepcopy(n_gram.phrasegrams)
                # Lets remove phrases that are with digits
                phrases_dict = {p: val for p, val in phrases_dict.items() if not any(char.isdigit() for char in p)}
                phrases_dict = filter_phrases(phrases_dict, phrases.vocab)
                n_gram.phrasegrams = phrases_dict
                sentences = [n_gram[s] for s in sentences]
                phrases_found.update(phrases_dict.keys())

        if nlp_adjust:
            sentences = nlp_adjustment(sentences)

        model = training_model(sentences)
    else:
        model = Word2Vec.load('word2vec.model')
    return model

# using the model, lets complete a sentence
# sentence = 'aang is the avatar and he is the last'

# cleaend_text = tokenize_text(
#     clean_paragraphs_text(
#         fetch_open_paragraph(
#             fetch_url('https://avatar.fandom.com/wiki/Aang')
#         )
#     )
# )