import tensorflow as tf

from model_context import ModelContext
import utils


def create_tflite(model=None):

    context = ModelContext.get_context()
    converter = \
        tf.lite.TFLiteConverter.from_saved_model(
            context.args.checkpoint_dir
        ) if model is None else \
        tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    return converter.convert()


def export_tflite(tflite_model):
    context = ModelContext.get_context()
    tflite_name = utils.flat_buffer_name(context.args)
    with open(tflite_name, 'wb') as f:
        f.write(tflite_model)
