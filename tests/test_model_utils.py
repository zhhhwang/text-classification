from text_classification.utils import model_utils
import os


def test_get_root_directory():
    test_path_name = model_utils.get_root_directory()
    assert os.path.basename(test_path_name) == 'text-classification'
