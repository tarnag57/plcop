import tensorflow as tf

from model_context import ModelContext

'''
Miscellaneous utility functions
'''


def print_dataset_stats(
    input_tensor_train,
    target_tensor_train,
    input_tensor_val,
    target_tensor_val
):
    print("\nDATA POINT STATS")
    print(f"Training input size:    {len(input_tensor_train)}")
    print(f"Training target size:   {len(target_tensor_train)}")
    print(f"Validation input size:  {len(input_tensor_val)}")
    print(f"Validation target size: {len(target_tensor_val)}")
    print()


def tensor_to_word_convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


def print_network_stats(encoder, decoder, train_dataset):
    context = ModelContext.get_context()

    # sample input
    example_input_batch, _example_target_batch = next(iter(train_dataset))
    sample_hidden = encoder.initialize_hidden_state()
    print(f"Input batch: {example_input_batch.shape}")
    print(f"Sample hidden: {sample_hidden.shape}")
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(
        sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(
        sample_hidden.shape))

    sample_decoder_output, _, _ = decoder(
        tf.random.uniform((context.args.batch_size, 1)),
        sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(
        sample_decoder_output.shape))

    print(encoder.summary())
    print(decoder.summary())


def shift_decoder_input(input_tensor):
    context = ModelContext.get_context()
    start_symbols = tf.repeat(
        [context.tokenizer.word_index['<start>']], input_tensor.shape[0])
    start_symbols = tf.reshape(start_symbols, [input_tensor.shape[0], 1])
    input_tensor_reduced = input_tensor[:, :-1]
    return tf.concat([start_symbols, input_tensor_reduced], axis=1)


def restore_checkpoint(checkpoint, checkpoint_dir=None):
    context = ModelContext.get_context()
    checkpoint_dir = context.args.checkpoint_dir if checkpoint_dir is None else checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def model_file_name(args):
    return args.save_dir + args.save_name + '.h5'


def lang_file_name(args):
    return args.save_dir + args.lang_name + '.json'


def save_model():
    context = ModelContext.get_context()

    # Writing the model
    tf.keras.models.save_model(
        context.seq_to_seq_model,
        model_file_name(context.args)
    )

    # Writing the generated language
    json_string = context.tokenizer.to_json()
    lang_file = open(lang_file_name(context.args), 'w')
    lang_file.write(json_string)
    lang_file.close()


def load_model(args):
    return tf.keras.models.load_model(model_file_name(args))


def load_lang(args):
    json_file = open(lang_file_name(args), 'r')
    json_string = json_file.read()
    return tf.keras.preprocessing.text.tokenizer_from_json(json_string)
