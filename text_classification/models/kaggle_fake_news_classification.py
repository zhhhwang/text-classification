from text_classification.data import get_dataset
from text_classification.config import constants
from text_classification.data import make_dataset
from text_classification.features import build_features
from text_classification.models import train_model
from text_classification.models import predict_model

from text_classification.config import fake_news_model_config

from sklearn.model_selection import train_test_split
import logging
import sys


def kaggle_fake_news_classification():
    """
    Main function that runs the kaggle fake news classification
    :return: None but could refer to logging info
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        stream=sys.stdout)

    # Get the data
    df = get_dataset.get_fake_news_data(constants.FAKE_NEWS_DATA_DIR)

    # Data preprocessing
    df = make_dataset.merge_column(df, ['text', 'title', 'subject'], 'content')
    del df['date']

    # Clean the dataset
    logging.info("Begin text denoising.")
    df['content'] = df['content'].apply(make_dataset.denoise_text)
    logging.info("Text denoising completed.")

    # Split training and testing
    x_train, x_test, y_train, y_test = train_test_split(df.content, df.category, random_state=0)

    # Text tokenizing and sequencing
    tokenizer = build_features.create_tokenizer_from_text(x_train, fake_news_model_config.MAX_FEATURES)
    x_train = build_features.create_text_sequence(x_train, tokenizer, fake_news_model_config.MAX_LEN)
    x_test = build_features.create_text_sequence(x_test, tokenizer, fake_news_model_config.MAX_LEN)

    # Getting glove embedding matrix
    embedding_matrix = build_features.get_embedding_matrix(build_features.get_glove_embedding_coef(fake_news_model_config.EMBEDDING_FILE),
                                                           tokenizer,
                                                           fake_news_model_config.MAX_FEATURES)

    # Create and train the model
    lstm_model = train_model.create_fake_news_model(embedding_matrix,
                                                    fake_news_model_config.MAX_FEATURES,
                                                    fake_news_model_config.MAX_LEN)
    _ = train_model.train_fake_news_model(lstm_model,
                                          x_train,
                                          y_train,
                                          x_test,
                                          y_test)
    _ = predict_model.get_confusion_matrix(lstm_model.predict_classes(x_test), y_test)


if __name__ == '__main__':
    kaggle_fake_news_classification()
