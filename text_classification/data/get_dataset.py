import pandas as pd
import logging


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


def get_amazon_review_data(data_dir):
    """
    Getting the data for the Amazon Review Project.
    :param data_dir:
    :return: Dataframe
    """
    df = pd.read_csv(data_dir + '/Reviews_small.csv')
    df = df[['Score', 'Summary', 'Text']]
    logging.info("Reading the dataset as of shape " + str(df.shape))

    return df
