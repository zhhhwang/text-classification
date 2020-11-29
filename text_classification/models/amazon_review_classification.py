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
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler(constants.LOG_FILE, 'w+'), logging.StreamHandler()])


def amazon_review_classification():
    """
    Main function that runs the kaggle fake news classification
    :return: None but could refer to logging info
    """

    # Get the data
    df = get_dataset.get_amazon_review_data(constants.AMAZON_REVIEW_DATA_DIR)

    # Data preprocessing
    df = make_dataset.merge_column(df, ['Summary', 'Text'], 'content')

    # Clean the dataset
    logging.info("Begin text denoising.")
    df['content'] = df['content'].apply(make_dataset.denoise_text)
    logging.info("Text denoising completed.")

    # The following words have low frequency, please consider replacing them when necessary.
    # model_utils.get_common_words(model_utils.get_word_frequency(df.content), -10)
    feature_count = model_utils.get_content_summary(df.content,
                                                    build_features.get_glove_embedding_coef(amazon_review_model_config.EMBEDDING_FILE),
                                                    least_common=50)
    content_word_count = model_utils.get_content_length_summary(df.content)

    # Split training and testing
    x_train, x_test, y_train, y_test = train_test_split(df.content, df.Score, random_state=0)

    # Modeling part
    glove_embedding_lstm_classification(x_train, x_test, y_train, y_test, feature_count,
                                        content_word_count)  # LSTM Classification
    glove_embedding_lstm_regression(x_train, x_test, y_train, y_test, feature_count,
                                    content_word_count)  # LSTM Regression
    bert_classification_model(x_train, x_test, y_train, y_test, content_word_count)  # BERT Classification


def glove_embedding_lstm_classification(x_train, x_test, y_train, y_test, feature_count, content_word_count):
    """
    lstm model for classification into 1 - 5 levels
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param feature_count: How many words are seen in the corpus and will be used to create the maximum feature count
    :param content_word_count: The 75 percentile of the content length * 1.2 and will be used for pad sequence
    :return: lstm model
    """
    # Text tokenizing and sequencing
    tokenizer = build_features.create_tokenizer_from_text(x_train, feature_count)
    x_train = build_features.create_text_sequence(x_train, tokenizer, content_word_count)
    x_test = build_features.create_text_sequence(x_test, tokenizer, content_word_count)
    y_train = build_features.get_one_hot_encoding(y_train)
    y_test = build_features.get_one_hot_encoding(y_test)

    # Getting glove embedding matrix
    embedding_matrix = build_features.get_embedding_matrix(
        build_features.get_glove_embedding_coef(amazon_review_model_config.EMBEDDING_FILE),
        tokenizer)

    # Create and train the model
    lstm_model = train_model.create_lstm_classification_model(embedding_matrix,
                                                              len(tokenizer.word_index) + 1,
                                                              content_word_count)
    _ = train_model.train_model(lstm_model, x_train, y_train, x_test, y_test, classification=True)
    print("Accuracy of the model on Training Data is - ", lstm_model.evaluate(x_train, y_train)[1] * 100, "%")
    print("Accuracy of the model on Testing Data is - ", lstm_model.evaluate(x_test, y_test)[1] * 100, "%")
    _ = predict_model.get_confusion_matrix(lstm_model.predict_classes(x_test), np.argmax(y_test, axis=1))

    return lstm_model


def glove_embedding_lstm_regression(x_train, x_test, y_train, y_test, feature_count, content_word_count):
    """

    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param feature_count:
    :param content_word_count:
    :return:
    """
    # Text tokenizing and sequencing
    tokenizer = build_features.create_tokenizer_from_text(x_train, feature_count)
    x_train = build_features.create_text_sequence(x_train, tokenizer, content_word_count)
    x_test = build_features.create_text_sequence(x_test, tokenizer, content_word_count)

    # Getting glove embedding matrix
    embedding_matrix = build_features.get_embedding_matrix(
        build_features.get_glove_embedding_coef(amazon_review_model_config.EMBEDDING_FILE),
        tokenizer)

    # Create and train the model
    lstm_model = train_model.create_lstm_regression_model(embedding_matrix,
                                                          len(tokenizer.word_index) + 1,
                                                          content_word_count)
    _ = train_model.train_model(lstm_model, x_train, y_train, x_test, y_test, classification=False)
    print("Accuracy of the model on Training Data is - ", lstm_model.evaluate(x_train, y_train))
    print("Accuracy of the model on Testing Data is - ", lstm_model.evaluate(x_test, y_test))

    return lstm_model


def bert_classification_model(x_train, x_test, y_train, y_test, content_word_count):
    """
    Create and train bert model
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param content_word_count:
    :return:
    """
    ids_train, masks_train, token_ids_train = train_model.create_bert_encoding(x_train, max_len=content_word_count)
    ids_test, masks_test, token_ids_test = train_model.create_bert_encoding(x_test, max_len=content_word_count)
    labels_train = build_features.get_one_hot_encoding(y_train)
    labels_test = build_features.get_one_hot_encoding(y_test)

    bert_model = train_model.create_bert_model(max_len=content_word_count)

    _ = train_model.train_bert_model(bert_model,
                                     (ids_train, masks_train, token_ids_train),
                                     labels_train,
                                     (ids_test, masks_test, token_ids_test),
                                     labels_test)
    print("Accuracy of the model on Training Data is - ", bert_model.evaluate((ids_train, masks_train, token_ids_train), labels_train)[1] * 100, "%")
    print("Accuracy of the model on Testing Data is - ", bert_model.evaluate((ids_test, masks_test, token_ids_test), labels_test)[1] * 100, "%")
    _ = predict_model.get_confusion_matrix(bert_model.predict_classes((ids_test, masks_test, token_ids_test)), np.argmax(y_test, axis=1))


if __name__ == '__main__':
    amazon_review_classification()
