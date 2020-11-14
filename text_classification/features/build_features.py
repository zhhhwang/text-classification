from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import logging


def tokenize_text(data, max_feature, max_len):
    """
    representing each word by a number
    :param data: A list of sentence
    :param max_feature: The number of most frequent words to keep
    :param max_len: Keep all the sample to have same length
    :return: Return the tokenizer and
    """
    logging.info("Creating text tokenizer.")
    tokenizer = Tokenizer(num_words=max_feature)
    tokenizer.fit_on_texts(data)
    tokenized_train = tokenizer.texts_to_sequences(data)
    return tokenizer, sequence.pad_sequences(tokenized_train, maxlen=max_len)


def get_embedding_word(word, *arr):
    """
    Get the word coefficient from the embedding file
    :param word:
    :param arr:
    :return:
    """
    return word, np.asarray(arr, dtype='float32')


def get_glove_embedding_coef(embedding_file):
    """
    Getting the pretrained embedding matrix from embedding file
    :param embedding_file:
    :return:
    """
    logging.info("Getting glove embedding coefficient.")
    return dict(get_embedding_word(*o.rstrip().rsplit(' ')) for o in open(embedding_file))


def get_embedding_matrix(embeddings_index,
                         tokenizer,
                         max_features):
    """
    Creating embedding matrix from glove embedding coefficients
    :param embeddings_index: embedding coefficients obtained from pre-tained file
    :param tokenizer: tokenizer created from corpus
    :param max_features: maximum features to be kept
    :return: embedding matrix to be put into the model
    """
    # Creating summarise embedding stats
    all_embedding = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embedding.mean(), all_embedding.std()
    embed_size = all_embedding.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))

    # Create the embedding matrix. Start with random normal
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    # For each words in the corpus, if located in the embedding index, then replace
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    logging.info("Embedding matrix created.")
    return embedding_matrix
