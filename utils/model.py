
"""Callable model function compatible with Experiment API.

Args:
    args: a Hargs object containing values for fields:
        F: A network mapping domain X to domain Y
        G: A network mapping domain Y to domain X
        discriminator_X: A binary classifier that distinguish elements in X
        discriminator_Y: A binary classifier that distinguish elements in Y
"""

import tensorflow as tf


class sequence_discriminator(tf.keras.Model):

    def __init__(self, args):
        super().__init__()
        filter_size = args.model.D.filter_size
        filter_count = args.model.D.filter_count
        self.dropout = args.model.D.dropout
        self.add_timing = args.model.add_timing
        self.args = args

        self.list_convs  =[]
        self.list_norms = []
        self.list_activations = []

        self.list_convs.append(Conv1D(
                dim_output=filter_size,
                filter_size=filter_count,
                stride=2))
        self.list_norms.append(lambda x: x)
        self.list_activations.append(lambda x: tf.maximum(x, 0.2 * x))

        for i in range(5):
            self.list_convs.append(Conv1D(
                    dim_output=filter_size * 2**(i + 1),
                    filter_size=filter_count,
                    stride=2))
            self.list_norms.append(tf.keras.layers.LayerNormalization())
            self.list_activations.append(lambda x: tf.maximum(x, 0.2 * x))

        self.list_convs.append(Conv1D(
                dim_output=1,
                filter_size=filter_count,
                stride=1))
        self.list_norms.append(lambda x: x)
        self.list_activations.append(lambda x: x)

    def call(self, x):
        if self.add_timing:
            x = timing(x, self.add_timing)

        if self.dropout != 0:
            x = tf.nn.dropout(x, 1 - self.dropout)

        for conv, norm, activation in zip(self.list_convs, self.list_norms, self.list_activations):
            x = conv(x)
            x = norm(x)
            x = activation(x)

        return x

class sequence_generator(tf.keras.Model):

    def __init__(self, args):
        super().__init__()
        filter_count = args.model.G.filter_count
        filter_size = args.model.G.filter_size
        self.add_timing = args.model.add_timing
        self.args = args

        self.list_layers = []
        self.list_resnet_convs = [[], []]
        self.list_activations = []

        self.list_layers.append(Conv1D(
                dim_output=filter_count,
                filter_size=filter_size,
                padding="valid"))
        self.list_activations.append(tf.keras.layers.ReLU())
        self.list_layers.append(Conv1D(
                dim_output=filter_count * 2,
                filter_size=filter_size))
        self.list_activations.append(tf.keras.layers.ReLU())
        self.list_layers.append(Conv1D(
                dim_output=filter_count * 4,
                filter_size=filter_size))
        self.list_activations.append(tf.keras.layers.ReLU())

        for i in range(5):
            self.list_resnet_convs[0].append(Conv1D(
                    dim_output=filter_count * 4,
                    filter_size=filter_size,
                    padding="valid"))
            self.list_resnet_convs[1].append(Conv1D(
                    dim_output=filter_count * 4,
                    filter_size=filter_size,
                    padding="valid"))

        self.list_layers.append(Conv1D(
                dim_output=args.vocab_size,
                filter_size=1,
                padding="valid"))

    def call(self, x):
        if self.add_timing:
            x = timing(x, self.add_timing)

        # x = tf.pad(x, [[0, 0], [self.args.model.G.filter_size // 2, self.args.model.G.filter_size // 2], [0, 0]], "CONSTANT")

        for conv, activation in zip(self.list_layers[:-1], self.list_activations):
            x = conv(x)
            x = activation(x)
        for conv_1, conv_2 in zip(*self.list_resnet_convs):
            x = build_resnet_block(x, conv_1, conv_2, self.args.model.G.filter_size, pad=False)
        x = self.list_layers[-1](x)

        output_dist = tf.nn.softmax(x)

        return output_dist


def monitor(X, X_hat, X_reconstruction, X_groundtruth, Y, Y_hat, Y_reconstruction, Y_groundtruth, vocab_size):
    """
    Ground truth loss logging. A metric for performance
    """
    X_mask = tf.cast(X>0, tf.float32)
    Y_mask = tf.cast(Y>0, tf.float32)
    X_groundtruth_acc = groundtruth_accuracy(
            tf.argmax(X_hat, axis=-1, output_type=tf.int32), X_groundtruth, Y_mask)
    Y_groundtruth_acc = groundtruth_accuracy(
            tf.argmax(Y_hat, axis=-1, output_type=tf.int32), Y_groundtruth, X_mask)

    X_groundtruth_dist = tf.one_hot(X_groundtruth, depth=vocab_size)
    Y_groundtruth_dist = tf.one_hot(Y_groundtruth, depth=vocab_size)

    X_groundtruth_loss = tf.reduce_sum(tf.reduce_sum(tf.abs(X_hat - X_groundtruth_dist), -1) * Y_mask, -1) / tf.reduce_sum(Y_mask, -1)
    X_groundtruth_loss = tf.reduce_mean(X_groundtruth_loss)
    Y_groundtruth_loss = tf.reduce_sum(tf.reduce_sum(tf.abs(Y_hat - Y_groundtruth_dist), -1) * X_mask, -1) / tf.reduce_sum(X_mask, -1)
    Y_groundtruth_loss = tf.reduce_mean(Y_groundtruth_loss)

    return X_groundtruth_loss, Y_groundtruth_loss, X_groundtruth_acc, Y_groundtruth_acc


def discrim_loss(X_discrim, Y_discrim, loss_type="wgan"):
    """Returns the discriminator loss.
    Args:
        X: Tensor from X domain to discriminate.
        Y: Tensor from Y domain to discriminate.
        train_towards_true: Bool. Train network to have discriminator recognize
            X and Y as true data samples.
    """
    if loss_type == "wgan":
        X_discrim_loss_to_True = -tf.reduce_mean(X_discrim)
        Y_discrim_loss_to_True = -tf.reduce_mean(Y_discrim)
        X_discrim_loss_to_False = -X_discrim_loss_to_True
        Y_discrim_loss_to_False = -Y_discrim_loss_to_True
    elif loss_type == "log":
        X_discrim_loss_to_True = tf.reduce_mean(tf.log(X_discrim))
        Y_discrim_loss_to_True = tf.reduce_mean(tf.log(Y_discrim))
        X_discrim_loss_to_False = -X_discrim_loss_to_True
        Y_discrim_loss_to_False = -Y_discrim_loss_to_True
    else:
        X_discrim_loss_to_True = tf.reduce_mean(tf.math.squared_difference(X_discrim, 1))
        Y_discrim_loss_to_True = tf.reduce_mean(tf.math.squared_difference(Y_discrim, 1))
        X_discrim_loss_to_False = tf.reduce_mean(X_discrim**2)
        Y_discrim_loss_to_False = tf.reduce_mean(Y_discrim**2)

    return (X_discrim_loss_to_True, Y_discrim_loss_to_True), (X_discrim_loss_to_False, Y_discrim_loss_to_False)


def cycle_gan_loss(G, F, D_X, D_Y, batch, args, W_emb=None):
    """Run CycleGAN on text."""
    dict_losses = {}
    # X, Y_ground_truth, Y, X_ground_truth = batch
    X, _, Y, _ = batch
    X_dist = tf.one_hot(X, depth=args.vocab_size)
    Y_dist = tf.one_hot(Y, depth=args.vocab_size)

    if W_emb:
        Y_hat = G(embed_inputs(X, W_emb))
        X_hat = F(embed_inputs(Y, W_emb))
        X_reconstruction = F(W_emb(Y_hat))
        Y_reconstruction = G(W_emb(X_hat))

        X_discrim = D_X(embed_inputs(X, W_emb))
        Y_discrim = D_Y(embed_inputs(Y, W_emb))
        X_hat_discrim = D_X(W_emb(X_hat))
        Y_hat_discrim = D_Y(W_emb(Y_hat))
    else:
        Y_hat = F(X_dist)
        X_hat = G(Y_dist)
        X_reconstruction = G(Y_hat)
        Y_reconstruction = F(X_hat)

        X_discrim = D_X(X_dist)
        Y_discrim = D_Y(Y_dist)
        X_hat_discrim = D_X(X_hat)
        Y_hat_discrim = D_Y(Y_hat)

    if args.model.lp_distance == "l0.5":
        X_reconstr_err = (X_dist - X_reconstruction)**0.5
        Y_reconstr_err = (Y_dist - Y_reconstruction)**0.5
    elif args.model.lp_distance == "l2":
        X_reconstr_err = (X_dist - X_reconstruction)**2
        Y_reconstr_err = (Y_dist - Y_reconstruction)**2
    elif args.model.lp_distance == "l1":
        X_reconstr_err = tf.abs(X_dist - X_reconstruction)
        Y_reconstr_err = tf.abs(Y_dist - Y_reconstruction)

    X_reconstr_err = tf.reduce_mean(tf.reduce_sum(X_reconstr_err, axis=2))
    Y_reconstr_err = tf.reduce_mean(tf.reduce_sum(Y_reconstr_err, axis=2))

    cycle_loss = X_reconstr_err + Y_reconstr_err

    (X_hat_discrim_loss_to_True, Y_hat_discrim_loss_to_True), (X_hat_discrim_loss_to_False, Y_hat_discrim_loss_to_False) = \
        discrim_loss(X_hat_discrim, Y_hat_discrim, loss_type=args.model.loss_type)

    F_loss = X_hat_discrim_loss_to_True + args.opti.cycle_loss * cycle_loss
    G_loss = Y_hat_discrim_loss_to_True + args.opti.cycle_loss * cycle_loss
    dict_losses['F_loss'] = F_loss
    dict_losses['G_loss'] = G_loss

    (X_discrim_loss_to_True, Y_discrim_loss_to_True), _ = \
        discrim_loss(X_discrim, Y_discrim, loss_type=args.model.loss_type)
    D_X_loss = X_discrim_loss_to_True + X_hat_discrim_loss_to_False
    D_Y_loss = Y_discrim_loss_to_True + Y_hat_discrim_loss_to_False

    if args.model.use_wasserstein:
        D_X_loss += wasserstein_penalty(D_X, X_dist, X_hat, W_emb, args)
        D_Y_loss += wasserstein_penalty(D_Y, Y_dist, Y_hat, W_emb, args)
    dict_losses['D_X_loss'] = D_X_loss
    dict_losses['D_Y_loss'] = D_Y_loss

    return dict_losses, (X_hat, Y_hat, X_reconstruction, Y_reconstruction)


def Conv1D(dim_output, filter_size, padding='same', stride=1):
    conv_op = tf.keras.layers.Conv1D(
        filters=dim_output,
        kernel_size=(filter_size,),
        strides=stride,
        padding=padding)

    return conv_op


# modified from https://github.com/hardikbansal/CycleGAN/blob/master/model.py
def build_resnet_block(x, Conv1D_1, Conv1D_2, filter_size, pad=True):
    if pad:
        out_res = tf.pad(x, [[0, 0]] + [[filter_size // 2, filter_size // 2]] *
                         (len(x.shape) - 2) + [[0, 0]], "REFLECT")
    else:
        out_res = x
    out_res = Conv1D_1(out_res)
    if pad:
        out_res = tf.pad(out_res, [[0, 0]] + [[filter_size // 2, filter_size // 2]] *
                         (len(x.shape) - 2) + [[0, 0]], "REFLECT")
    out_res = Conv1D_2(out_res)

    return tf.nn.relu(out_res + x)


def timing(x, timing_type):

    if timing_type == "transformer":
        shape = x.shape
        num = tf.reshape(tf.range(shape[1]), [1, -1, 1])
        num = tf.cast(tf.tile(num, [shape[0], 1, shape[2]]), tf.float32)
        denom = tf.reshape(tf.range(shape[2]), [1, 1, -1])
        denom = tf.tile(denom, [shape[0], shape[1], 1])
        denom = tf.cast(10000**(denom / shape[2]), tf.float32)

        sine_timing = tf.sin(num / denom)
        cos_timing = tf.cos(num / denom)
        layerings = tf.tile([[[True, False]]], [shape[0], shape[1], shape[2] // 2])
        timing = tf.where(layerings, sine_timing, cos_timing)

        return x + timing

    # elif args.timing_type == "concat":
    #     timing = tf.get_variable(
    #             "timing",
    #             shape=[args.sample_length, args.hidden_size],
    #             dtype=tf.float32,
    #             initializer=tf.random_normal_initializer(
    #                     mean=0.0, stddev=1.0))
    #     timing = tf.tile(tf.expand_dims(timing, 0), [args.batch_size, 1, 1])
    #     timing = timing[:tf.shape(x)[0], :tf.shape(x)[1], :]
    #     return tf.concat([x, timing], 2)
    # else:
    #     raise Exception("Bad timing type %s" % args.timing_type)

def embed_inputs(inputs, W_emb):
    return tf.gather(W_emb.variables[0], inputs)


def softmax_to_embedding(x, W_emb):
    vocab_size, hidden_size = W_emb.variables[0].shape
    o_shape = x.shape.as_list()
    if o_shape[0] is None:
        o_shape[0] = tf.shape(x)[0]
    if o_shape[1] is None:
        o_shape[1] = tf.shape(x)[1]

    output_dist = tf.reshape(x, [o_shape[0] * o_shape[1], vocab_size])
    output = W_emb(output_dist)
    output = tf.reshape(output, [o_shape[0], o_shape[1], hidden_size])

    return output


# def construct_vocab_lookup_table(vocab):
#     mapping_string = tf.constant(vocab)
#     return tf.contrib.lookup.index_to_string_table_from_tensor(
#             mapping_string, default_value="<UNK>")


# def log_text(F, G, args):
#     lookup_table = construct_vocab_lookup_table(args.vocab)
#
#     X_vocab = tf.expand_dims(tf.range(args.vocab_size), axis=0)
#     if args.model.use_embeddings:
#         X = embed_inputs(X_vocab, args, reuse=True)
#     else:
#         X = tf.one_hot(X_vocab, depth=args.vocab_size)
#     X_map_distribution = F(X, args.F, args)
#     X_map_indices = tf.argmax(X_map_distribution, axis=-1)
#     X_map_text = lookup_table.lookup(tf.to_int64(X_map_indices))
#
#     X_vocab_text = lookup_table.lookup(tf.to_int64(X_vocab))
#     X_text = tf.string_join([X_vocab_text, "->", X_map_text])
#     tf.summary.text("F_map", X_text)
#
#     Y_vocab = tf.expand_dims(tf.range(args.vocab_size), axis=0)
#     if args.model.use_embeddings:
#         Y = embed_inputs(Y_vocab, args, reuse=True)
#     else:
#         Y = tf.one_hot(Y_vocab, depth=args.vocab_size)
#     Y_map_distribution = G(Y, args.G, args)
#     Y_map_indices = tf.argmax(Y_map_distribution, axis=-1)
#     Y_map_text = lookup_table.lookup(tf.to_int64(Y_map_indices))
#
#     Y_vocab_text = lookup_table.lookup(tf.to_int64(Y_vocab))
#     Y_text = tf.string_join([Y_vocab_text, "->", Y_map_text])
#     tf.summary.text("G_map", Y_text)


def groundtruth_accuracy(A, A_groundtruth, mask):
    groundtruth_equalities = tf.cast(tf.equal(A, A_groundtruth), tf.float32)
    groundtruth_accs = tf.reduce_sum(groundtruth_equalities * mask, axis=1) / tf.reduce_sum(mask, axis=1)

    return tf.reduce_mean(groundtruth_accs)


def sample_along_line(A_true, A_fake):
    batch_size, len_seq, size_hidden = A_fake.shape

    A_unif = tf.tile(tf.random.uniform([batch_size, 1, 1]),
                     [1, len_seq, size_hidden])

    return A_unif * A_fake + (1 - A_unif) * A_true


def wasserstein_penalty(discriminator, A_true, A_fake, W_emb, args):
    A_interp = sample_along_line(A_true, A_fake)
    if args.model.use_embeddings:
        A_interp = W_emb(A_interp)
    with tf.GradientTape() as t:
        t.watch(A_interp)
        discrim_A_interp = discriminator(A_interp)
    discrim_A_grads = t.gradient(discrim_A_interp, A_interp)
    discrim_A_grads = tf.squeeze(discrim_A_grads)

    if args.model.original_l2:
        l2_loss = tf.sqrt(tf.reduce_sum(discrim_A_grads**2, axis=[1, 2]))
        loss = args.model.wasserstein_loss * tf.reduce_mean((l2_loss - 1)**2)
    else:
        loss = args.model.wasserstein_loss * (tf.nn.l2_loss(discrim_A_grads) - 1)**2

    return loss
