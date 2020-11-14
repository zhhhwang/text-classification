import pandas as pd
import logging
from text_classification.config import constants


def get_fake_news_data(data_dir):
    """
    Getting the data and concatenate
    :param data_dir:
    :return:
    """
    # Reading the file
    true = pd.read_csv(data_dir + '/true.csv')
    false = pd.read_csv(data_dir + '/fake.csv')

    # Setting the label
    true['category'] = 1
    false['category'] = 0

    df = pd.concat([true, false])
    logging.info("Reading the dataset as of shape " + str(df.shape))

    return df

