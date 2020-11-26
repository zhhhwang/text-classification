from text_classification.data import get_dataset
from text_classification.config import constants
from text_classification.data import make_dataset
from text_classification.features import build_features
from text_classification.models import train_model
from text_classification.models import predict_model
from text_classification.utils import model_utils

from text_classification.config import amazon_review_model_config

from sklearn.model_selection import train_test_split
import logging
import sys
import numpy as np


def amazon_review_classification():
    """
    Main function that runs the kaggle fake news classification
    :return: None but could refer to logging info
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        stream=sys.stdout)

    # Get the data
    df = get_dataset.get_amazon_review_data(constants.AMAZON_REVIEW_DATA_DIR)

    # Data preprocessing
    df = make_dataset.merge_column(df, ['Summary', 'Text'], 'content')

    # Clean the dataset
    logging.info("Begin text denoising.")
    df['content'] = df['content'].apply(make_dataset.denoise_text)
    logging.info("Text denoising completed.")

    # The following words have low frequency, please consider replacing them when necessary.
    model_utils.get_common_words(model_utils.get_word_frequency(df.content), -10)
    feature_count = model_utils.get_content_summary(df.content,
                                                    build_features.get_glove_embedding_coef(amazon_review_model_config.EMBEDDING_FILE),
                                                    least_common=50)
    content_word_count = model_utils.get_content_length_summary(df.content)

    # Split training and testing
    x_train, x_test, y_train, y_test = train_test_split(df.content, df.Score, random_state=0)

    # Text tokenizing and sequencing
    tokenizer = build_features.create_tokenizer_from_text(x_train, feature_count)
    x_train = build_features.create_text_sequence(x_train, tokenizer, content_word_count)
    x_test = build_features.create_text_sequence(x_test, tokenizer, content_word_count)
    y_train = build_features.get_one_hot_encoding(y_train)
    y_test = build_features.get_one_hot_encoding(y_test)

    # Getting glove embedding matrix
    embedding_matrix = build_features.get_embedding_matrix(build_features.get_glove_embedding_coef(amazon_review_model_config.EMBEDDING_FILE),
                                                           tokenizer,
                                                           feature_count)

    # Create and train the model
    lstm_model = train_model.create_lstm_classification_model(embedding_matrix, feature_count, content_word_count)
    _ = train_model.train_model(lstm_model, x_train, y_train, x_test, y_test)
    _ = predict_model.get_confusion_matrix(lstm_model.predict_classes(x_test), np.argmax(y_test, axis=1))


if __name__ == '__main__':
    amazon_review_classification()
