from text_classification.config import amazon_review_model_config
import keras
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import logging
from transformers import BertTokenizer


def create_lstm_classification_model(embedding_matrix, max_features, max_length):
    """
    Create LSTM model
    :param embedding_matrix: embedding matrix created from glove
    :param max_features: maximum feature numbers
    :param max_length: maximum length of the news
    :return: LSTM mode to be trained
    """
    # Defining Neural Network
    model = Sequential()

    # Non-trainable embedding layer
    model.add(Embedding(max_features,
                        output_dim=amazon_review_model_config.EMBED_SIZE,
                        weights=[embedding_matrix],
                        input_length=max_length,
                        trainable=False))

    # LSTM Model
    model.add(Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.25, dropout=0.25)))
    model.add(Bidirectional(LSTM(units=64, recurrent_dropout=0.1, dropout=0.1)))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(lr=amazon_review_model_config.LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    logging.info(model.summary())
    return model


def create_lstm_regression_model(embedding_matrix, max_features, max_length):
    """
    Create LSTM model
    :param embedding_matrix: embedding matrix created from glove
    :param max_features: maximum feature numbers
    :param max_length: maximum length of the news
    :return: LSTM mode to be trained
    """
    # Defining Neural Network
    model = Sequential()

    # Non-trainable embedding layer
    model.add(Embedding(max_features,
                        output_dim=amazon_review_model_config.EMBED_SIZE,
                        weights=[embedding_matrix],
                        input_length=max_length,
                        trainable=False))

    # LSTM Model
    model.add(Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.25, dropout=0.25)))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(units=64, recurrent_dropout=0.1, dropout=0.1)))
    model.add(Dropout(0.1))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(lr=amazon_review_model_config.LEARNING_RATE),
                  loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    logging.info(model.summary())
    return model


def create_bert_model(max_len=128):
    """

    :param max_len:
    :return:
    """
    ids = tf.keras.layers.Input(shape=(max_len, ), dtype=tf.int32)
    masks = tf.keras.layers.Input(shape=(max_len, ), dtype=tf.int32)
    token_ids = tf.keras.layers.Input(shape=(max_len, ), dtype=tf.int32)

    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
    pooled_output, sequence_output = bert_layer([ids, masks, token_ids])
    output = sequence_output[:, 0, :]
    out = tf.keras.layers.Dropout(0.5)(output)
    out = tf.keras.layers.Dense(5, activation='softmax')(out)

    model = tf.keras.models.Model(inputs=[ids, masks, token_ids], outputs=out)
    optimizer = tf.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    logging.info(model.summary())
    return model


def create_bert_encoding(df, max_len):
    """
    Create bert encoding for a dataframe
    :param df:
    :param max_len: maximum length of the sentence
    :return:
    """
    # Create bert tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    ids, masks, token_ids = zip(*df.apply(lambda x: bert_processing_data(x, tokenizer, max_len=max_len)))

    ids = np.array(ids, dtype='float32')
    masks = np.array(masks, dtype='float32')
    token_ids = np.array(token_ids, dtype='float32')

    return ids, masks, token_ids


def bert_processing_data(row, tokenizer, max_len):
    """
    Preprocess the data for bert encoding. (Row label will be used as is)
    :param row:
    :param tokenizer:
    :param max_len:
    :return:
    """
    temp_input_ids = tokenizer.encode(row, max_length=max_len)
    pad_len = max_len - len(temp_input_ids)
    input_ids = temp_input_ids + [0] * pad_len
    attention_masks = [1] * len(temp_input_ids) + [0] * pad_len
    token_type_ids = [0] * max_len

    return np.array(input_ids), np.array(attention_masks), np.array(token_type_ids)


def train_model(model, x_train, y_train, x_test, y_test, classification=True):
    """

    :param model:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param classification:
    :return:
    """
    logging.info("Model training start.")

    # Setting callbacks based on classification/regression
    if classification:
        callbacks = [amazon_review_model_config.classification_early_stopping,
                     amazon_review_model_config.classification_learning_rate_reduction]
    else:
        callbacks = [amazon_review_model_config.regression_early_stopping,
                     amazon_review_model_config.regression_learning_rate_reduction]

    model_log = model.fit(x_train,
                          y_train,
                          batch_size=amazon_review_model_config.BATCH_SIZE,
                          validation_data=(x_test, y_test),
                          epochs=amazon_review_model_config.EPOCHS,
                          callbacks=callbacks)
    logging.info("Model training ends.")

    return model_log


def train_bert_model(model, x_train, y_train, x_test, y_test):
    """
    Train bert model
    :param model:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    logging.info("Model training start.")

    callbacks = [amazon_review_model_config.classification_early_stopping,
                 amazon_review_model_config.classification_learning_rate_reduction]

    model_log = model.fit(x_train,
                          y_train,
                          batch_size=amazon_review_model_config.BATCH_SIZE,
                          validation_data=(x_test, y_test),
                          epochs=amazon_review_model_config.EPOCHS,
                          callbacks=callbacks)
    logging.info("Model training ends.")

    return model_log
