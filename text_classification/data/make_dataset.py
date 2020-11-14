from bs4 import BeautifulSoup
import re
import logging


def merge_column(data, col_list):
    """
    Merge the designated column and create a new column containing all text info
    :param  data: input dataset
    :param  col_list: list of string identifying columns to be merged. Columns in the list
                      will be removed
    :return: a dataset.
    """

    logging.info("Merging " + str(col_list) + " to column text.")
    data['text'] = data[col_list[0]]
    del data[col_list[0]]
    col_list.pop(0)
    for column in col_list:
        data['text'] = ['%s %s' % x for x in zip(data['text'], data[column])]
        del data[column]

    return data


def strip_html(text):
    """
    Provide strip html format
    :param text: html texts
    :return: text stripped
    """
    logging.info("Stripping html.")
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    """
    Removing square brackets
    :param text: text containing square brackets
    :return: text removed square brackets
    """
    logging.info("Removing [] related content from text.")
    return re.sub('\[[^]]*\]', '', text)


def remove_url(text):
    """
    Removing urls
    :param text: text containing urls (anything start with a http)
    :return: text removed urls
    """
    logging.info("Removing urls.")
    return re.sub(r'http\S+', '', text)


def remove_stopwords(text, stop_words):
    """
    Remove stop words
    :param text: input text
    :param stop_words: list of stop words
    :return:
    """
    logging.info("Removing stop words.")
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop_words:
            final_text.append(i.strip())
    return " ".join(final_text)


def denoise_text(text):
    """
    Apply text removal functions
    :param text: input text
    :return: processed text
    """
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_url(text)
    text = remove_stopwords(text)
    return text
