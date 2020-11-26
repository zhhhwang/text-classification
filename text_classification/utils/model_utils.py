from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import logging


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
        result = counter.most_common(n)
    else:
        result = counter.most_common()[:n - 1:-1]

    logging.info("Please see the following most/least seen word in the comments.")
    print(result)
    return result


def get_content_summary(content, embedding, least_common=20):
    """
    Get the content summary on the word coverage from the embedding
    :param content: a pandas series
    :param embedding: embedding matrix
    :param least_common: number of least common word to show
    :return: count of the total words in the corpus
    """
    count = 0
    uncovered_words = {}
    covered_words = {}
    for text in content:
        text = text.split()
        for word in text:
            if word not in embedding.keys():
                count += 1
                if word not in uncovered_words:
                    uncovered_words[word] = 1
                else:
                    uncovered_words[word] += 1
            else:
                if word not in covered_words:
                    covered_words[word] = 1
                else:
                    covered_words[word] += 1
    print("---There are {} words in the whole dataset, and {:.2f}% of the words aren't covered by Glove---".format(
        (len(uncovered_words) + len(covered_words)),
        len(uncovered_words) / (len(uncovered_words) + len(covered_words)) * 100))
    print('---Top 20 most common uncovered words---')

    if least_common is not None:
        print(pd.DataFrame([uncovered_words]).T.reset_index().sort_values(by=0, ascending=False).head(least_common))

    return len(uncovered_words) + len(covered_words)


def get_content_length_summary(content):
    """
    Get a descriptive summary of the length of the content
    :param content: pd series
    :return: The median of the content length
    """
    logging.info("Please refer to the following summary statistics on the comments length after cleaning.")
    result = content.apply(lambda x: len(x.split())).describe()
    print(result)

    return int(np.ceil(result['75%']) * 1.2)
