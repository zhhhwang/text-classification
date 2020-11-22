from text_classification.config import amazon_review_model_config
import keras
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM

import logging


def create_amazon_review_model(embedding_matrix,
                               max_features,
                               max_length):
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
    model.add(LSTM(units=128, return_sequences=True, recurrent_dropout=0.25, dropout=0.25))
    model.add(LSTM(units=64, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(lr=amazon_review_model_config.LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_amazon_review_model(model, x_train, y_train, x_test, y_test):
    """
    Train the model
    :param model: LSTM Model
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: model log
    """
    logging.info("Model training start.")
    model_log = model.fit(x_train,
                          y_train,
                          batch_size=amazon_review_model_config.BATCH_SIZE,
                          validation_data=(x_test, y_test),
                          epochs=amazon_review_model_config.EPOCHS,
                          callbacks=[amazon_review_model_config.learning_rate_reduction])
    logging.info("Model training ends.")
    print("Accuracy of the model on Training Data is - ", model.evaluate(x_train, y_train)[1] * 100, "%")
    print("Accuracy of the model on Testing Data is - ", model.evaluate(x_test, y_test)[1] * 100, "%")

    return model_log
