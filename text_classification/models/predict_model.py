from sklearn.metrics import confusion_matrix


def get_confusion_matrix(y_test, pred):
    """

    :param pred:
    :param y_test:
    :return:
    """
    matrix = confusion_matrix(y_test, pred)
    print(matrix)

    return matrix
