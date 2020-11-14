from keras.callbacks import ReduceLROnPlateau

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