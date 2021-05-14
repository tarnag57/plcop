import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import utils

import dataset
import models
import model_compression
from model_context import ModelContext
import params
import predict
import preprocess
import training


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

    # Model Save/Load
    parser.add_argument(
        '--save_dir',
        default=params.SAVE_DIR,
        help='The directory of the final output.',
        type=str,
    )
    parser.add_argument(
        '--save_name',
        default=params.SAVE_NAME,
        help='The name of the model file.',
        type=str,
    )
    parser.add_argument(
        '--load_name',
        default=params.LOAD_NAME,
        help='The name of the model file.',
        type=str,
    )
    parser.add_argument(
        '--lang_name',
        default=params.LANG_NAME,
        help='The name of the language file.',
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

    # Optimisation
    parser.add_argument(
        '--pruning',
        help='Enable pruning during training',
        type=bool
    )
    parser.add_argument(
        '--quant_aware',
        help='Enable quantisation aware training',
        type=bool
    )

    return parser


def init_context(prediction_phase=False, load_data=True, load_tokenizer=False):
    # In prediction phase, we load the saved model and language tokenizer
    # and do NOT use the training datset

    parser = build_parser()
    args = parser.parse_args()

    tokenizer = None
    model = None

    input_tensor = None
    target_tensor = None

    # Loading / Constructing the tokenizer
    if prediction_phase:
        model = utils.load_model(args)

    if load_tokenizer:
        tokenizer = utils.load_lang(args)

    if load_data:
        input_tensor, target_tensor, tokenizer = preprocess.load_dataset(
            args.path_to_file,
            args.num_examples,
            args.max_length,
            tokenizer
        )

    # Loading / Constructing the tokenizer
    if prediction_phase:
        tokenizer = utils.load_lang(args)
        model = utils.load_model(args)

    vocab_size = len(tokenizer.word_index) + 1

    if prediction_phase:
        model = utils.load_model(args)
    else:
        model = models.build_models(
            vocab_size,
            args.embedding_dim,
            args.units
        )

    optimizer = tf.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(model)

    ModelContext.create_context(
        args=args,
        optimizer=optimizer,
        checkpoint=checkpoint,                    # TODO
        tokenizer=tokenizer,
        seq_to_seq_model=model
    )

    # Finally create the dataset
    if load_data:

        # Creating training and validation sets using an 80-20 split
        train_input, val_input, train_target, val_target = train_test_split(
            input_tensor, target_tensor, test_size=0.2)

        # Creating datasets
        train_dataset = preprocess.create_dataset_from_tensor(
            train_input,
            train_target,
            args.batch_size,
            args.buffer_size_mult
        )
        val_dataset = preprocess.create_dataset_from_tensor(
            val_input,
            val_target,
            args.batch_size,
            args.buffer_size_mult
        )
        ModelContext.add_datset(
            input_tensor, train_dataset, train_input, val_dataset, val_input)


def main():

    # For consistency throughout test runs
    tf.random.set_seed(987654)

    init_context(prediction_phase=True, load_tokenizer=True)
    context = ModelContext.get_context()
    print(f"Training example shape: {context.train_input[0].shape}")
    # utils.restore_checkpoint(context.checkpoint, context.args.checkpoint_dir)
    models.lstm_training(context.seq_to_seq_model)

    # print(f"Trining is complete, saving the model...")
    # utils.save_model()

    # training.perform_training()

    # print(context.encoder.summary())
    # utils.save_model(output_dir='./saved_models/')
    # tflite = model_compression.create_tflite()

    # utils.restore_checkpoint(context.checkpoint)
    # clause = "16 [-(k7_xcmplx_0(VAR,VAR)=k7_xcmplx_0(VAR,VAR)), VAR=VAR]"
    # clause = "51 [v1_xboole_0(u1_struct_0(SKLM)), m1_subset_1(u1_struct_0(SKLM),k1_zfmisc_1(u1_struct_0(SKLM))), v12_waybel_0(u1_struct_0(SKLM),SKLM), v1_waybel_0(u1_struct_0(SKLM),SKLM)]"
    # result = predict.seq_to_seq_predict(clause)
    # print(result)
    # tf.print("Num GPUs Available: ", len(
    #     tf.config.list_physical_devices('GPU')))

    # init_context(prediction_phase=True, load_tokenizer=True)
    # context = ModelContext.get_context()
    # context.seq_to_seq_model.summary()
    clause = "51 [k7_partfun1(VAR,VAR,VAR)=k1_funct_1(VAR,VAR)]"
    enc_out, enc_hidden = predict.encode_clause(clause)
    result = predict.decode_clause(enc_out, enc_hidden)
    print(result)

    # tf.keras.models.save_model(context.encoder, './saved_model/encoder')
    # tf.keras.models.save_model(context.decoder, './saved_model/decoder')
    # res = preprocess.preprocess_sentence(
    #     "14 [-(k3_xcmplx_0(VAR,VAR)=k3_xcmplx_0(VAR,VAR)), VAR=VAR]")
    # print(res)

    # init_context(prediction_phase=True, load_tokenizer=True)
    # context = ModelContext.get_context()
    # encoder = models.get_encoder_part(context.seq_to_seq_model)
    # encoder.summary()
    # tflite = model_compression.create_tflite(encoder)
    # model_compression.export_tflite(tflite)


if __name__ == "__main__":
    main()
