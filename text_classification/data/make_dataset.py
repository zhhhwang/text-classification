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


def remove_contractions(text):
    """
    Removing all the contractions.
    :param text: input text string
    :return: output text
    """
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"We're", "We are", text)
    text = re.sub(r"That's", "That is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"Can't", "Cannot", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"What's", "What is", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"There's", "There is", text)
    text = re.sub(r"He's", "He is", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"You're", "You are", text)
    text = re.sub(r"I'M", "I am", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"i'm", "I am", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"Isn't", "is not", text)
    text = re.sub(r"Here's", "Here is", text)
    text = re.sub(r"you've", "you have", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"y'all", "you all", text)
    text = re.sub(r"would've", "would have", text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"We've", "We have", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"Y'all", "You all", text)
    text = re.sub(r"Weren't", "Were not", text)
    text = re.sub(r"Didn't", "Did not", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"DON'T", "DO NOT", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"should've", "should have", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"They're", "They are", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"I'd", "I would", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"i've", "I have", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"I've", "I have", text)
    text = re.sub(r"Don't", "Do not", text)
    text = re.sub(r"I'll", "I will", text)
    text = re.sub(r"Let's", "Let us", text)
    text = re.sub(r"you'd", "You would", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"Ain't", "am not", text)
    text = re.sub(r"Haven't", "Have not", text)
    text = re.sub(r"Could've", "Could have", text)
    text = re.sub(r"youve", "you have", text)

    return text


def remove_punctuations(text):
    """
    Actually this is not removing but separate punctuations.
    :param text:
    :return:
    """
    punctuations = ',@#!?+&*[]-%.:/();$=><|{}^' + "'`" + "\"" + "\'"
    for p in punctuations:
        text = text.replace(p, f' {p} ')

    return text


def denoise_text(text):
    """
    Apply text removal functions
    :param text: input text
    :return: processed text
    """
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_url(text)
    text = remove_contractions(text)
    text = remove_punctuations(text)
    text = remove_stopwords(text, get_stop_words())
    return text
