from text_classification.features import build_features
import numpy as np


def test_tokenize_text():
    test_doc = ['Well done!',
                'Good work',
                'Great effort',
                'nice work',
                'Excellent!']

    _, test_doc_vector = build_features.tokenize_text(test_doc,
                                                      max_feature=10,
                                                      max_len=2)
    assert test_doc_vector.shape == (5, 2)
    print(test_doc_vector)
    assert (test_doc_vector == np.array([[2, 3], [4, 1], [5, 6], [7, 1], [0, 8]])).all()
