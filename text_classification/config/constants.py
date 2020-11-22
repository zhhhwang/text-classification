from text_classification.utils import model_utils

# Directory
LOCAL_DATASETS_DIR = model_utils.get_root_directory() + '/data'

# Datasets
FAKE_NEWS_DATA_DIR = LOCAL_DATASETS_DIR + '/interim/'
AMAZON_REVIEW_DATA_DIR = LOCAL_DATASETS_DIR + '/interim/'
EMBEDDING_FILE_DIR = LOCAL_DATASETS_DIR + '/external/'

# Glove Vec dataset url
GLOVE_WIKI = 'http://nlp.stanford.edu/data/glove.6B.zip'
GLOVE_TWITTER = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
