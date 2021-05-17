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
        default='./saved_models/newly_trained/u-128-pruning.tflite',
        help="The filename for the inference model.",
        metavar='model',
        type=str
    )

    parser.add_argument(
        '--lang',
        '-l',
        default='./saved_models/newly_trained/len-300-lang.json',
        help="The filename for the model language.",
        metavar='lang',
        type=str
    )

    parser.add_argument(
        '--log_file',
        default='./server_log/server_run0.txt',
        help='Metadata generated during the running of the server is logged here.',
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
