from text_classification.models.predict_model import get_confusion_matrix
import numpy as np


def test_get_confusion_matrix():
    y_test = np.asarray([1, 1, 2, 3])
    pred_test = np.asarray([2, 1, 2, 3])

    matrix = get_confusion_matrix(y_test, pred_test)
    expected = np.asarray([[1, 1, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

    assert (matrix == expected).all()
