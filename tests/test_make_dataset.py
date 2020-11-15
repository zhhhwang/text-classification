from text_classification.data import make_dataset
import pandas as pd
from tests import sample_data


def test_pandas_merge_column():
    test_data = pd.DataFrame({'text1': ['a', 'b'], 'text2': ['c', 'd'], 'numeric': [1, 2]})

    test_data_result = make_dataset.merge_column(test_data, ['text1', 'text2'], 'text')

    # Check if text1 is removed
    try:
        test_data_result['text1']
        assert False
    except KeyError:
        assert True

    # Check if text 2 is removed
    try:
        test_data_result['text2']
        assert False
    except KeyError:
        assert True

    assert all(test_data_result['text'] == ['a c', 'b d'])
    assert all(test_data_result['numeric'] == [1, 2])


def test_remove_between_square_brackets():
    assert make_dataset.remove_between_square_brackets(sample_data.TEST_SENTENCE) == \
           'This is a test from http:12321.com and it  brackets like .'


def test_remove_url():
    assert make_dataset.remove_url(sample_data.TEST_SENTENCE) == \
           'This is a test from  and it [contains] brackets like [].'


def test_remove_stopwords():
    test_stop_words = ['this', 'it', 'from']
    assert make_dataset.remove_stopwords(sample_data.TEST_SENTENCE, test_stop_words) == \
           'is a test http:12321.com and [contains] brackets like [].'
