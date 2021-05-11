'''
Default params
'''

# Data Parameters
MAX_LENGTH = 300
NUM_EXAMPLES = None
PATH_TO_FILE = './data/training_input.txt'

# Model Parameters
EMBEDDING_DIM = 256
UNITS = 256

# Training Parameters
BUFFER_SIZE_MULT = 1
BATCH_SIZE = 256
EPOCHS = 2

# Checkpointing
CHECKPOINT_FREQ = 4
CHECKPOINT_DIR = './new-u-256/training_checkpoints'
CHECKPOINT_PREFIX = 'ckpt'

# Model Save/Load Params
SAVE_DIR = './new-u-256/saved_models'
SAVE_NAME = 'model'
LANG_NAME = 'lang'

# Prediction
PRED_MAX_LEN = MAX_LENGTH
