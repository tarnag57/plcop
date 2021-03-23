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


def restore_checkpoint(checkpoint, checkpoint_dir=None):
    context = ModelContext.get_context()
    checkpoint_dir = context.args.checkpoint_dir if checkpoint_dir is None else checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
