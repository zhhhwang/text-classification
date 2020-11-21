from text_classification.data import get_dataset
from text_classification.config import constants
from text_classification.data import make_dataset
from text_classification.features import build_features
from text_classification.models import train_model
from text_classification.models import predict_model

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

    # Split training and testing
    x_train, x_test, y_train, y_test = train_test_split(df.content, df.Score, random_state=0)

    # Text tokenizing and sequencing
    tokenizer = build_features.create_tokenizer_from_text(x_train, amazon_review_model_config.MAX_FEATURES)
    x_train = build_features.create_text_sequence(x_train, tokenizer, amazon_review_model_config.MAX_LEN)
    x_test = build_features.create_text_sequence(x_test, tokenizer, amazon_review_model_config.MAX_LEN)
    y_train = build_features.get_one_hot_encoding(y_train)
    y_test = build_features.get_one_hot_encoding(y_test)

    # Getting glove embedding matrix
    embedding_matrix = build_features.get_embedding_matrix(build_features.get_glove_embedding_coef(amazon_review_model_config.EMBEDDING_FILE),
                                                           tokenizer,
                                                           amazon_review_model_config.MAX_FEATURES)

    # Create and train the model
    lstm_model = train_model.create_amazon_review_model(embedding_matrix,
                                                        amazon_review_model_config.MAX_FEATURES,
                                                        amazon_review_model_config.MAX_LEN)
    _ = train_model.train_amazon_review_model(lstm_model,
                                              x_train,
                                              y_train,
                                              x_test,
                                              y_test)
    _ = predict_model.get_confusion_matrix(lstm_model.predict_classes(x_test), np.argmax(y_test, axis=1))


if __name__ == '__main__':
    amazon_review_classification()
