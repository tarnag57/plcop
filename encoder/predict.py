import tensorflow as tf
import numpy as np

import models
import preprocess
from model_context import ModelContext


def preprocess_clause(clause, context, numbered=True):
    clause = preprocess.preprocess_sentence(clause, numbered)
    tokenized = preprocess.tokenize(clause, context.tokenizer)
    vocab_size = len(context.tokenizer.word_index) + 1
    return tf.one_hot(tokenized, depth=vocab_size)


def encode_clause(clause, numbered=True):
    context = ModelContext.get_context()
    input_tensor = preprocess_clause(clause, context, numbered)
    print(input_tensor)

    model = models.get_encoder_part(context.seq_to_seq_model)
    print(f"Encoder input shape: {input_tensor.shape}")
    encoder_states = model.predict(input_tensor)

    return encoder_states[-1]


def decode_clause(clause, encoder_states, max_len=200, numbered=True):
    context = ModelContext.get_context()
    model = models.get_decoder_part(
        context.seq_to_seq_model, context.args.units)

    # Empty target sequence
    vocab_size = len(context.tokenizer.word_index) + 1
    target = np.zeros((1, 1, vocab_size))
    target[0, 0, context.tokenizer.word_index['<start>']] = 1.0

    hidden_states = [encoder_states[i].reshape(
        1, len(encoder_states[i])) for i in range(2)]

    sequence = ""
    should_stop = False
    while not should_stop:
        decoder_inputs = [target, hidden_states[0], hidden_states[1]]
        output, h, c = model.predict(decoder_inputs)

        sampled_index = np.argmax(output[0, -1, :])
        sampled_token = "@" if sampled_index == 0 \
            else context.tokenizer.index_word[sampled_index]
        sequence += sampled_token + " "

        if sampled_token == "<end>" or len(sequence) > max_len:
            should_stop = True

        target = np.zeros((1, 1, vocab_size))
        target[0, 0, sampled_index] = 1.0
        hidden_states = [h, c]

    return sequence


def seq_to_seq_predict(clause, max_len=200, numbered=True):
    encoder_states = encode_clause(clause, numbered=numbered)
    return decode_clause(
        clause,
        encoder_states,
        max_len=200,
        numbered=numbered
    )


# def encode_clause(clause):

#     context = ModelContext.get_context()
#     clause = preprocess.preprocess_sentence(clause)

#     inputs = [context.inp_lang.word_index[i] for i in clause.split(' ')]
#     inputs = tf.keras.preprocessing.sequence.pad_sequences(
#         [inputs],
#         maxlen=context.args.pred_max_len,
#         padding='post'
#     )
#     inputs = tf.convert_to_tensor(inputs)

#     hidden = [tf.zeros((1, context.args.units))]
#     enc_out, enc_hidden = context.encoder(inputs, hidden)

#     return enc_out, enc_hidden


# def decode_clause(enc_out, hidden, show_attention=False):

#     context = ModelContext.get_context()
#     result = ''
#     attention_plot = np.zeros(
#         (context.args.pred_max_len, context.args.pred_max_len)) if show_attention else None

#     dec_hidden = hidden
#     dec_input = tf.expand_dims([context.targ_lang.word_index['<start>']], 0)

#     for t in range(context.args.pred_max_len):
#         predictions, dec_hidden, attention_weights \
#             = context.decoder(dec_input,
#                               dec_hidden,
#                               enc_out)

#         # storing the attention weights to plot later on
#         if not attention_plot is None:
#             attention_weights = tf.reshape(attention_weights, (-1, ))
#             attention_plot[t] = attention_weights.numpy()

#         predicted_id = tf.argmax(predictions[0]).numpy()

#         result += context.targ_lang.index_word[predicted_id] + ' '

#         if context.targ_lang.index_word[predicted_id] == '<end>':
#             return result, attention_plot

#         # the predicted ID is fed back into the model
#         dec_input = tf.expand_dims([predicted_id], 0)

#     return result, attention_plot
