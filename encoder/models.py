import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_schedule, pruning_callbacks
from tensorflow_model_optimization.python.core.quantization.keras import quantize
import os

from model_context import ModelContext
import utils


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# def build_gru_encoder(vocab_size, embedding_dim, enc_units, batch_sz):
#     encoder_input = layers.Input(shape=(None,))
#     encoder_embed = layers.Embedding(vocab_size, embedding_dim)(encoder_input)
#     output, state = tf.keras.layers.GRU(
#         enc_units,
#         return_sequences=True,
#         return_state=True,
#         recurrent_initializer='glorot_uniform'
#     )(encoder_embed)


def build_models(vocab_size, embedding_dim, enc_units):

    encoder_input = layers.Input(shape=(None, vocab_size))

    # Note: encoder_output == state_h
    _enc_output, state_h, state_c = layers.LSTM(
        enc_units, return_state=True)(encoder_input)
    encoder_states = [state_h, state_c]

    decoder_input = tf.keras.Input(shape=(None, vocab_size))
    decoder_output, _, _ = layers.LSTM(
        enc_units,
        return_sequences=True,
        return_state=True
    )(decoder_input, initial_state=encoder_states)

    decoder_output = layers.Dense(
        vocab_size, activation="softmax")(decoder_output)

    print(f"Model output shape: {decoder_output.shape}")

    seq_to_seq_model = tf.keras.Model(
        [encoder_input, decoder_input], decoder_output)

    return seq_to_seq_model


def get_encoder_part(seq_to_seq_model):
    encode_inputs = seq_to_seq_model.input[0]
    encoder_outputs, state_h, state_c = seq_to_seq_model.layers[2].output
    return tf.keras.models.Model(encode_inputs, [state_h, state_c])


def get_decoder_part(seq_to_seq_model, units):
    decoder_input = seq_to_seq_model.input[1]
    state_input_h = tf.keras.Input(shape=(units,), name="input_3")
    state_input_c = tf.keras.Input(shape=(units,), name="input_4")

    decoder_lstm = seq_to_seq_model.layers[3]
    decoder_output, state_h, state_c = decoder_lstm(
        decoder_input, initial_state=[state_input_h, state_input_c]
    )

    decoder_dense = seq_to_seq_model.layers[4]
    decoder_output = decoder_dense(decoder_output)

    return tf.keras.models.Model(
        [decoder_input, state_input_h, state_input_c],
        [decoder_output, state_h, state_c]
    )


def lstm_training(model):

    context = ModelContext.get_context()
    checkpoint_prefix = os.path.join(
        context.args.checkpoint_dir, context.args.checkpoint_prefix)
    train_log_name = checkpoint_prefix + "-train.txt"

    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    logger = tf.keras.callbacks.CSVLogger(
        train_log_name, separator=',', append=False)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     context.args.checkpoint_dir,
    #     save_freq=context.args.checkpoint_freq
    # )
    # callbacks = [logger, checkpoint]
    callbacks = [logger]

    if context.args.pruning:
        print("Pruning enabled")
        model = prune.prune_low_magnitude(model, pruning_schedule.PolynomialDecay(
            initial_sparsity=0.3,
            final_sparsity=0.7,
            begin_step=1000,
            end_step=3000
        ))
        # It is important that we do the optimization steps before logging and checkpointing
        callbacks = [pruning_callbacks.UpdatePruningStep()] + callbacks

    if context.args.quant_aware:
        print("Quant-aware training enabled")
        model = quantize.quantize_model(model)

    model.compile(
        optimizer="adam", loss=loss_function, metrics=['accuracy'])

    model.summary()

    vocab_size = len(context.tokenizer.word_index) + 1
    train_dataset = utils.generate_model_datset(
        context.train_input, vocab_size, context.args.batch_size)
    val_dataset = utils.generate_model_datset(
        context.val_input, vocab_size, context.args.batch_size)

    model.fit(
        x=train_dataset,
        batch_size=context.args.batch_size,
        validation_data=val_dataset,
        epochs=context.args.epochs,
        callbacks=callbacks
    )


# def build_decoder(vocab_size, embedding_dim, enc_units, timesteps):
#     decoder_input = layers.RepeatVector(timesteps)()
#     decoder_input = layers.Input(shape=(None,))
#     decoder_embed = layers.Embedding(vocab_size, embedding_dim)(decoder_input)
#     decoder_output = layers.LSTM(decoder_embed, initial_state=)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x):
        print(x.shape)
        hidden = self.initialize_hidden_state(x.shape[0])
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    # In order to perform prediction, we need to make the batch_size variable
    def initialize_hidden_state(self, batch_sz):
        return tf.zeros((batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights
