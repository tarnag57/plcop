import tensorflow as tf

from model_context import ModelContext


def create_tflite():
    context = ModelContext.get_context()
    converter = tf.lite.TFLiteConverter.from_saved_model(
        context.args.checkpoint_dir)
    return converter.convert()
