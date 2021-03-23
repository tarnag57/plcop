import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import argparse
import numpy as np
import io
import time

import dataset
import params
import predict
import preprocess
import training
import utils
from model_context import ModelContext
from models import Encoder, Decoder


def build_parser():
    parser = argparse.ArgumentParser(
        description='Run the training process.'
    )

    # Data Parameters
    parser.add_argument(
        '--max_length',
        '-l',
        default=params.MAX_LENGTH,
        help='The maximum length of training samples.',
        metavar='len',
        type=int,
    )
    parser.add_argument(
        '--num_examples',
        '-n',
        default=params.NUM_EXAMPLES,
        help='The number of training samples. Set to None to include the whole set.',
        metavar='num',
        type=int,
    )
    parser.add_argument(
        '--path_to_file',
        '-f',
        default=params.PATH_TO_FILE,
        help='The file contatining the training samples.',
        metavar='file',
        type=str,
    )

    # Model Parameters
    parser.add_argument(
        '--embedding_dim',
        '-d',
        default=params.EMBEDDING_DIM,
        help='The dimensionality of the encoder output.',
        metavar='dim',
        type=int,
    )
    parser.add_argument(
        '--units',
        '-u',
        default=params.UNITS,
        help='The dimensionility of the output of each RNN cell.',
        metavar='units',
        type=int,
    )

    # Training Parameters
    parser.add_argument(
        '--buffer_size_mult',
        default=params.BUFFER_SIZE_MULT,
        help='The buffer size compared to the dataset size.',
        metavar='buff',
        type=float,
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        default=params.BATCH_SIZE,
        help='The number of samples in a batch.',
        metavar='batch',
        type=int,
    )
    parser.add_argument(
        '--epochs',
        '-e',
        default=params.EPOCHS,
        help='The number of epochs to be performed.',
        metavar='epochs',
        type=int,
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint_freq',
        default=params.CHECKPOINT_FREQ,
        help='The frequency of performing checkpointing.',
        metavar='freq',
        type=int,
    )
    parser.add_argument(
        '--checkpoint_dir',
        default=params.CHECKPOINT_DIR,
        help='The directory to which the checkpoints are written to.',
        metavar='ckpt_dir',
        type=str,
    )
    parser.add_argument(
        '--checkpoint_prefix',
        default=params.CHECKPOINT_PREFIX,
        help='The prefix used for each checkpoint.',
        metavar='ckpt_prefix',
        type=str,
    )

    # Prediction
    parser.add_argument(
        '--pred_max_len',
        default=params.PRED_MAX_LEN,
        help='The maximum length for prediction.',
        metavar='pred_len',
        type=int,
    )

    return parser


def init_context():

    parser = build_parser()
    args = parser.parse_args()

    # Need to know the input / target language size
    # Loading the dataset
    input_tensor, target_tensor, inp_lang, targ_lang = preprocess.load_dataset(
        args.path_to_file,
        args.num_examples,
        args.max_length
    )

    # Creating dataset
    train_dataset, val_dataset, train_size, val_size = preprocess.create_datasets(
        input_tensor,
        target_tensor,
        args.batch_size,
        args.buffer_size_mult
    )

    # Calculate steps per epoch
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    steps_per_epoch = train_size // args.batch_size
    val_steps_per_epoch = val_size // args.batch_size

    encoder = Encoder(
        vocab_inp_size,
        args.embedding_dim,
        args.units,
        args.batch_size
    )
    decoder = Decoder(
        vocab_tar_size,
        args.embedding_dim,
        args.units,
        args.batch_size
    )

    optimizer = tf.keras.optimizers.Adam()

    # Checkpointing config
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    encoder = ModelContext.create_context(
        args,
        encoder,
        decoder,
        optimizer,
        train_dataset,
        train_size,
        val_dataset,
        val_size,
        checkpoint,
        inp_lang,
        targ_lang,
        steps_per_epoch,
        val_steps_per_epoch
    )


def main():
    init_context()
    context = ModelContext.get_context()

    training.perform_training()

    # utils.restore_checkpoint(context.checkpoint)
    # clause = "51 [v1_xboole_0(u1_struct_0(SKLM)), m1_subset_1(u1_struct_0(SKLM),k1_zfmisc_1(u1_struct_0(SKLM))), v12_waybel_0(u1_struct_0(SKLM),SKLM), v1_waybel_0(u1_struct_0(SKLM),SKLM)]"

    # enc_out, enc_hidden = predict.encode_clause(clause)
    # result = predict.decode_clause(enc_out, enc_hidden)
    # print(result)

    # res = preprocess.preprocess_sentence(
    #     "14 [-(k3_xcmplx_0(VAR,VAR)=k3_xcmplx_0(VAR,VAR)), VAR=VAR]")
    # print(res)


if __name__ == "__main__":
    main()
