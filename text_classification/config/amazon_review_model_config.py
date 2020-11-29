from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from text_classification.config import constants

# Data feature
EMBEDDING_FILE = constants.EMBEDDING_FILE_DIR + 'glove.twitter.27B.100d.txt'

# Model Params
BATCH_SIZE = 256
EPOCHS = 20
EMBED_SIZE = 100
LEARNING_RATE = 0.01

# Call Backs
classification_learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                           patience=2,
                                                           verbose=1,
                                                           factor=0.5,
                                                           min_lr=0.000001)

classification_early_stopping = EarlyStopping(monitor='val_accuracy',
                                              min_delta=0,
                                              patience=3,
                                              verbose=1,
                                              mode='auto')

# Call Backs
regression_learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                       patience=2,
                                                       verbose=1,
                                                       factor=0.5,
                                                       min_lr=0.000001)

regression_early_stopping = EarlyStopping(monitor='val_accuracy',
                                          min_delta=0,
                                          patience=3,
                                          verbose=1,
                                          mode='auto')
