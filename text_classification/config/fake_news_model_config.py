from keras.callbacks import ReduceLROnPlateau
from text_classification.config import constants

# Data feature
MAX_FEATURES = 10000
MAX_LEN = 300
EMBEDDING_FILE = constants.EMBEDDING_FILE_DIR + 'glove.twitter.27B.100d.txt'

# Model Params
BATCH_SIZE = 256
EPOCHS = 10
EMBED_SIZE = 100
LEARNING_RATE = 0.01

# Call Back
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)