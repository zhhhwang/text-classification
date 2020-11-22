from text_classification.utils import model_utils
import collections
import pandas as pd


def test_get_root_directory():
    test_path_name = model_utils.get_root_directory()

    assert type(test_path_name) == str


def test_get_word_frequency():
    test_data = {'test_text': ['This', 'This', 'is', 'a', 'test']}
    test_pd_df = pd.DataFrame(data=test_data)
    test_word_freq = model_utils.get_word_frequency(test_pd_df.test_text)

    assert type(test_word_freq) == collections.Counter
    assert test_word_freq['This'] == 2


def test_get_common_words():
    test_data = {'test_text': ['this', 'this', 'this', 'is', 'is', 'a', 'test']}
    test_pd_df = pd.DataFrame(data=test_data)
    test_count = model_utils.get_word_frequency(test_pd_df.test_text)
    expect_most_common = [('this', 3), ('is', 2)]
    expect_least_common = [('a', 1), ('test', 1)]

    assert all([i in model_utils.get_common_words(test_count, -2) for i in expect_least_common])
    assert all([i in model_utils.get_common_words(test_count, 2) for i in expect_most_common])
