import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        description='Run the inference server process.'
    )

    parser.add_argument(
        '--addr',
        '-a',
        default="127.0.0.1",
        help="The address on which the server is listening.",
        type=str
    )

    parser.add_argument(
        '--port',
        '-p',
        default=2042,
        help="The port on which the server is run.",
        metavar='port',
        type=int
    )

    parser.add_argument(
        '--model',
        '-m',
        default='./u-256/saved_models/model.tflite',
        help="The filename for the inference model.",
        metavar='model',
        type=str
    )

    parser.add_argument(
        '--lang',
        '-l',
        default='./u-256/saved_models/lang.json',
        help="The filename for the model language.",
        metavar='lang',
        type=str
    )

    return parser
