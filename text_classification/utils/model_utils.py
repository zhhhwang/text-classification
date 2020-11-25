from pathlib import Path
from collections import Counter


def get_root_directory():
    """
    return the project path
    :return: Path to the project root directory. This is a Path object
    """
    return str(Path(__file__).parent.parent.parent)


def get_word_frequency(data):
    """
    Getting the word frequency of the data
    :param data: text column of a pandas dataframe
    :return: counter and count of the text
    """
    words = []
    for i in data:
        for j in i.split():
            words.append(j.strip())

    return Counter(words)


def get_common_words(counter, n=10):
    """
    Get the most common words in a corpus
    :param counter: a counter object that counts the corpus
    :param n: top n will return. It would either be a int or list
    :return: a dictionary that shows the most common words
    """
    if n > 0:
        return counter.most_common(n)
    else:
        return counter.most_common()[:n - 1:-1]
