import tensorflow as tf
import time
import os

from model_context import ModelContext


def perform_training():

    context = ModelContext.get_context()
    checkpoint_prefix = os.path.join(
        context.args.checkpoint_dir, context.args.checkpoint_prefix)
    tain_log_name = checkpoint_prefix + "-train.txt"
    val_log_name = checkpoint_prefix + "-val.txt"
    train_log_file = open(tain_log_name, "w")
    val_log_file = open(val_log_name, "w")

    print("Building train step")
    train_step = construct_train_step(
        context.encoder,
        context.decoder,
        context.optimizer,
        context.targ_lang
    )

    for epoch in range(context.args.epochs):
        start = time.time()

        total_loss = 0

        for (batch, (inp, targ)) in enumerate(
            context.train_dataset.take(context.steps_per_epoch)
        ):
            print("starting training for batch")
            batch_loss = train_step(inp, targ)
            total_loss += batch_loss

            if batch % 2 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % context.args.checkpoint_freq == 0:
            context.checkpoint.save(file_prefix=checkpoint_prefix)

        # Calculate overall epoch loss
        normalised_tain_loss = total_loss / context.steps_per_epoch
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, normalised_tain_loss))
        train_log_file.write(str(normalised_tain_loss))
        train_log_file.flush()

        # Perform validation step
        val_loss = perform_validation()
        normalised_val_loss = val_loss / context.val_steps_per_epoch
        print('Epoch {} Val Loss {:.4f}'.format(
            epoch + 1, normalised_val_loss))
        val_log_file.write(str(normalised_val_loss))
        val_log_file.flush()

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / context.steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # Cleaning up log files
    val_log_file.close()
    train_log_file.close()


def perform_validation():

    context = ModelContext.get_context()
    val_step = construct_validation_step(
        context.encoder, context.decoder, context.targ_lang)

    total_loss = 0
    for (_batch, (inp, targ)) in enumerate(
        context.val_dataset.take(context.steps_per_epoch)
    ):
        batch_loss = val_step(inp, targ)
        total_loss += batch_loss

    return total_loss


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def construct_validation_step(encoder, decoder, targ_lang):

    context = ModelContext.get_context()

    @ tf.function
    def val_step(inp, targ):
        loss = 0

        enc_output, enc_hidden = encoder(inp)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims(
            [targ_lang.word_index['<start>']] * context.args.batch_size, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(
                dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        return batch_loss

    return val_step


def construct_train_step(encoder, decoder, optimizer, targ_lang):

    context = ModelContext.get_context()

    @ tf.function
    def train_step(inp, targ):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp)
            print("Encoder calculated")

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims(
                [targ_lang.word_index['<start>']] * context.args.batch_size, 1)

            print(f"Going through decoder, targ shape: {targ.shape}")
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                print(t)
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(
                    dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    return train_step
