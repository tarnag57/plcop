'''
Default params
'''

# Data Parameters
MAX_LENGTH = 300
NUM_EXAMPLES = None
PATH_TO_FILE = './data/training_input.txt'

# Model Parameters
EMBEDDING_DIM = 256
UNITS = 128

# Training Parameters
BUFFER_SIZE_MULT = 1
BATCH_SIZE = 8
EPOCHS = 1

# Checkpointing
CHECKPOINT_FREQ = 4
CHECKPOINT_DIR = './u-128-pruning/training_checkpoints'
CHECKPOINT_PREFIX = 'ckpt'

# Model Save/Load Params
SAVE_DIR = './saved_models/newly_trained'
SAVE_NAME = 'u-128-final'
LOAD_NAME = 'u-128-final'
LANG_NAME = 'len-300-lang'

# Prediction
PRED_MAX_LEN = MAX_LENGTH
