import tensorflow as tf

from model_context import ModelContext


def create_dataset(
    x_tensor,
    y_tensor,
    vocab_inp_size,
    vocab_tar_size,
    batch_size,
    steps_per_epoch,
    buffer_size=None,
    embedding_dim=256,
    units=1024
):
    context = ModelContext.get_context()
    if buffer_size is None:
        buffer_size = len(x_tensor) * context.args.buffer_size_mult

    dataset = tf.data.Dataset.from_tensor_slices(
        (x_tensor, y_tensor)).shuffle(buffer_size)
    return dataset.batch(batch_size, drop_remainder=True)
