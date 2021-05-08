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

    parser.add_argument(
        '--log_requests',
        default=None,
        help="The file name to which the requests are logged. If None, no logging takes palce",
        type=str
    )

    parser.add_argument(
        '--empty_response',
        default=None,
        help="Send a constant 0 vector (with given dimension) response rather than running the model",
        type=int
    )

    return parser
