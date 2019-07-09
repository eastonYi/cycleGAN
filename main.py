import tensorflow as tf
from tensorflow.keras.layers import Dense
from utils.model import sequence_generator, sequence_discriminator, cycle_gan_loss, monitor
from utils.dataset import load_data, load_vocab, cipher_generator
from time import time
from utils.arguments import args
import numpy as np

np.set_printoptions(precision=3)


def main(args):
    """The basic CycleGAN template.

    Args:
        features: a dict containing key "X" and key "Y"
        mode: training, evaluation or infer
    """
    # load dataset
    train_data, test_data, plain_vocab = cipher_generator(vocab_path=args.dirs.vocab, output_dir='data')
    args.vocab_size = len(plain_vocab)
    tfdata_train = tf.data.Dataset.from_generator(lambda: train_data,
        (tf.int32, tf.int32), (tf.TensorShape([None]), tf.TensorShape([None])))
    iter_train = iter(tfdata_train.cache().\
        repeat().shuffle(1000).padded_batch(args.batch_size, ([args.max_seq_len], [args.max_seq_len])).prefetch(buffer_size=5))

    tfdata_test = tf.data.Dataset.from_generator(lambda: test_data,
        (tf.int32, tf.int32), (tf.TensorShape([None]), tf.TensorShape([None])))
    iter_test = iter(tfdata_test.cache().padded_batch(args.batch_size, ([args.max_seq_len], [args.max_seq_len])).prefetch(buffer_size=5))

    # gather transformations and descriminators
    F = sequence_generator(args)
    G = sequence_generator(args)
    D_X = sequence_discriminator(args)
    D_Y = sequence_discriminator(args)
    W_emb = Dense(args.model.dim_hidden, use_bias=False)
    W_emb(tf.zeros([1, args.vocab_size]))

    optimizer_G = tf.keras.optimizers.Adam(args.opti.G.lr, beta_1=0.9, beta_2=0.999)
    optimizer_D = tf.keras.optimizers.Adam(args.opti.D.lr, beta_1=0.9, beta_2=0.999)

    def train_step(batch, losses, optimizer, variables):
        """
        train_gX, G_X_loss, generator_G
        train_gY, G_Y_loss, generator_F
        train_dX, D_X_loss, D_X
        train_dY, D_Y_loss, D_Y
        train_emb, embedding_loss
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(variables)
            dict_losses, predicts = cycle_gan_loss(
                    G, F, D_X, D_Y, batch, args=args, W_emb=W_emb)
            loss = tf.reduce_sum([dict_losses[type] for type in losses])

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return dict_losses, predicts

    for iteration in range(99999):
        start = time()

        X, Y_groundtruth = next(iter_train)
        X_groundtruth, Y = next(iter_train)
        batch = [X, Y_groundtruth, Y, X_groundtruth]
        if iteration == 0:
            cycle_gan_loss(G, F, D_X, D_Y, batch, args=args, W_emb=W_emb)
            F.summary(); G.summary(); D_X.summary(); D_Y.summary()
            var_Gs = F.trainable_variables + G.trainable_variables + W_emb.trainable_variables
            var_Ds = D_X.trainable_variables + D_Y.trainable_variables + W_emb.trainable_variables

        if iteration % args.opti.D_G_rate != 0:
            dict_losses, predicts = train_step(batch, ['D_X_loss', 'D_Y_loss'], optimizer_D, var_Ds)
        if iteration % args.opti.D_G_rate == 0:
            dict_losses, predicts = train_step(batch, ['G_loss', 'F_loss'], optimizer_G, var_Gs)

        X_hat, Y_hat, X_reconstruction, Y_reconstruction = predicts

        X_groundtruth_loss, Y_groundtruth_loss, X_groundtruth_acc, Y_groundtruth_acc = \
            monitor(X, X_hat, X_reconstruction, X_groundtruth, Y, Y_hat, Y_reconstruction, Y_groundtruth, args.vocab_size)

        if iteration % 10 == 0:
            print('G:{:.3f}\t F:{:.3f}\t D_X:{:.2f}\tD_Y:{:.2f}\tlabel loss: {:.2f}|{:.2f}\t label acc: {:.3f}|{:.3f} batch:{} used: {:.2f} iter: {}'.format(
                   dict_losses['G_loss'], dict_losses['F_loss'], dict_losses['D_X_loss'], dict_losses['D_Y_loss'], X_groundtruth_loss, Y_groundtruth_loss, X_groundtruth_acc, Y_groundtruth_acc,
                   X.shape, time()-start, iteration))


if __name__ == '__main__':
    from argparse import ArgumentParser
    import os

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default='')
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
    print('enter the TRAINING phrase')

    main(args)