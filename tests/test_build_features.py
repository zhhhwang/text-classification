from text_classification.features import build_features
import numpy as np


def test_create_text_sequence():
    test_doc = ['Well done!',
                'Good work',
                'Great effort',
                'nice work',
                'Excellent!']

    tokenizer = build_features.create_tokenizer_from_text(test_doc, 10)
    test_doc_vector = build_features.create_text_sequence(test_doc,
                                                          tokenizer,
                                                          max_len=2)
    assert test_doc_vector.shape == (5, 2)
    assert (test_doc_vector == np.array([[2, 3], [4, 1], [5, 6], [7, 1], [0, 8]])).all()
