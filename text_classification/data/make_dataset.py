from bs4 import BeautifulSoup
import re
import logging
from nltk.corpus import stopwords
import string


def merge_column(data, col_list, new_name):
    """
    Merge the designated column and create a new column containing all text info
    :param  data: input dataset
    :param  col_list: list of string identifying columns to be merged. Columns in the list
                      will be removed
    :param new_name: the new name of the merged columns
    :return: a dataset.
    """

    logging.info("Merging " + str(col_list) + " to column text.")
    data[new_name] = data[col_list[0]]
    del data[col_list[0]]
    col_list.pop(0)
    for column in col_list:
        data[new_name] = ['%s %s' % x for x in zip(data[new_name], data[column])]
        del data[column]

    return data


def strip_html(text):
    """
    Provide strip html format
    :param text: html texts
    :return: text stripped
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    """
    Removing square brackets
    :param text: text containing square brackets
    :return: text removed square brackets
    """
    return re.sub(r'\[[^]]*\]', '', text)


def remove_url(text):
    """
    Removing urls
    :param text: text containing urls (anything start with a http)
    :return: text removed urls
    """
    return re.sub(r'http\S+', '', text)


def remove_stopwords(text, stop_words):
    """
    Remove stop words
    :param text: input text
    :param stop_words: list of stop words
    :return:
    """
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop_words:
            final_text.append(i.strip().lower())
    return " ".join(final_text)


def get_stop_words():
    """
    Get english stop words and adding string punctuations
    :return:
    """
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)

    return stop


def denoise_text(text):
    """
    Apply text removal functions
    :param text: input text
    :return: processed text
    """
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_url(text)
    text = remove_stopwords(text, get_stop_words())
    return text
