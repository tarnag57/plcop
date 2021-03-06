import tensorflow as tf
import unicodedata
import re
import io


'''
Utilities for creating and pre-processing a dataset
'''


def create_dataset_from_tensor(
    input_tensor,
    target_tensor,
    batch_size,
    buffer_size_mult
):
    buffer_size = len(input_tensor) * buffer_size_mult
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor, target_tensor)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def load_dataset(path, num_examples, max_length, tokenizer=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_raw_dataset(path, num_examples, max_length)

    if tokenizer is None:
        tokenizer = create_tokenizer(inp_lang)

    input_tensor = tokenize(inp_lang, tokenizer)
    target_tensor = tokenize(targ_lang, tokenizer)

    return input_tensor, target_tensor, tokenizer


def unicode_to_ascii(s):
    # Converts the unicode file to ascii
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


# Numbered indicates whether the input data has a line number as the first token
def preprocess_sentence(w, numbered=False):
    w = unicode_to_ascii(w.lower().strip())

    # Remove the number at the beginning of the line
    first_token = 1 if numbered else 0
    w = " ".join(w.split(' ')[first_token:])

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r'([.,!?()=\-\[\]])', r' \1 ', w)
    w = re.sub(r'\s{2,}', ' ', w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z0-9_(),=\-\[\]]+", " ", w)

    w = w.strip()

    # adding a start an end token to the sentence so that the model know
    # when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def num_tokens(w):
    return len(w.split(' '))


def create_raw_dataset(path, num_examples, max_length):

    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    #    For autoencoding (same lang), it will be [WORDS, WORDS]
    lines = io.open(path).read().strip().split('\n')[:num_examples]

    # Since we are using autoencoding, we need to duplicate the word
    word_pairs = [[preprocess_sentence(w) for _ in range(2)] for w in lines]

    print(f"Original number of word_pairs: {len(word_pairs)}")
    word_pairs = [a for a in word_pairs if num_tokens(
        a[0]) <= max_length]
    print(f"Filtered number of word_pairs: {len(word_pairs)}")

    return zip(*word_pairs)


def create_tokenizer(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)
    return lang_tokenizer


def tokenize(input_tokens, tokenizer):
    tensor = tokenizer.texts_to_sequences(input_tokens)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    return tensor


def _test_preprocessing():
    # Quickly test preprocessing
    en_sentence = u"May I borrow this book?"
    sp_sentence = u"¿Puedo tomar prestado este libro?"
    print(preprocess_sentence(en_sentence))
    print(preprocess_sentence(sp_sentence).encode('utf-8'))


def _test_dataset_compilation(path_to_file):
    # Testing dataset compilation
    en, sp = create_raw_dataset(path_to_file, None, max_length=100)
    print(en[-1])
    print(sp[-1])
