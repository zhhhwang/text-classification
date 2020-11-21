from pathlib import Path


def get_root_directory():
    """
    return the project path
    :return: Path to the project root directory. This is a Path object
    """
    return Path(__file__).parent.parent.parent


def get_word_frequency():
    return None


def get_glove_embedding_file():
    return None
