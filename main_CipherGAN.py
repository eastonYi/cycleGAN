import tensorflow as tf
from tensorflow.keras.layers import Dense
from models.seq_model import sequence_generator, sequence_discriminator, cycle_gan_loss, monitor, mini
from utils.dataset import cipher_generator
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


    m, table = mini(args)
    x = tf.ones([args.batch_size, args.max_seq_len, 28])

    with tf.GradientTape(persistent=True) as tape:
        loss = m(x)
    gradients = tape.gradient(loss, m.trainable_variables)
    import pdb; pdb.set_trace()

    # gather transformations and descriminators
    F, F_time_table = sequence_generator(args)
    G, G_time_table = sequence_generator(args)
    D_X, DX_time_table = sequence_discriminator(args)
    D_Y, DY_time_table = sequence_discriminator(args)
    F.summary(); G.summary(); D_X.summary(); D_Y.summary()
    W_emb = Dense(args.model.dim_hidden, use_bias=False)
    W_emb(tf.zeros([1, args.vocab_size]))

    optimizer_G = tf.keras.optimizers.Adam(args.opti.G.lr, beta_1=0.9, beta_2=0.999)
    optimizer_F = tf.keras.optimizers.Adam(args.opti.G.lr, beta_1=0.9, beta_2=0.999)
    optimizer_DX = tf.keras.optimizers.Adam(args.opti.D.lr, beta_1=0.9, beta_2=0.999)
    optimizer_DY = tf.keras.optimizers.Adam(args.opti.D.lr, beta_1=0.9, beta_2=0.999)
    optimizer_Embed = tf.keras.optimizers.Adam(args.opti.G.lr, beta_1=0.9, beta_2=0.999)


    ckpt = tf.train.Checkpoint(G=G,
                               F=F,
                               D_X=D_X,
                               D_Y=D_Y,
                               optimizer_G=optimizer_G,
                               optimizer_F=optimizer_F,
                               optimizer_DX=optimizer_DX,
                               optimizer_DY=optimizer_DY,
                               optimizer_Embed = optimizer_Embed)

    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    #     print ('Latest checkpoint restored!!')

    # @tf.function
    def train_step(batch):
        """
        train_gX, G_X_loss, generator_G
        train_gY, G_Y_loss, generator_F
        train_dX, D_X_loss, D_X
        train_dY, D_Y_loss, D_Y
        train_emb, embedding_loss
        """

        with tf.GradientTape(persistent=True) as tape:
            dict_losses, predicts = cycle_gan_loss(
                    G, F, D_X, D_Y, batch, args=args, W_emb=W_emb)

        for loss_type, vars, optimizer in zip(
            ['G_loss', 'F_loss', 'D_X_loss', 'D_Y_loss', 'Embed_loss'],
            [G.trainable_variables+[G_time_table], F.trainable_variables+[F_time_table], D_X.trainable_variables+[DX_time_table], D_Y.trainable_variables+[DY_time_table], W_emb.trainable_variables],
            [optimizer_G, optimizer_F, optimizer_DX, optimizer_DY, optimizer_Embed]):
            import pdb; pdb.set_trace()
            gradients = tape.gradient(dict_losses[loss_type], vars)
            optimizer.apply_gradients(zip(gradients, vars))
        del tape

        return dict_losses, predicts

    for iteration in range(99999):
        start = time()

        X, Y_groundtruth = next(iter_train)
        X_groundtruth, Y = next(iter_train)
        batch = [X, Y_groundtruth, Y, X_groundtruth]

        dict_losses, predicts = train_step(batch)

        X_hat, Y_hat, X_reconstruction, Y_reconstruction = predicts
        _, _, X_groundtruth_acc, Y_groundtruth_acc = \
            monitor(X, X_hat, X_reconstruction, X_groundtruth, Y, Y_hat, Y_reconstruction, Y_groundtruth, args.vocab_size)

        if iteration % 50 == 0:
            print('G:{:.3f}\t  F:{:.3f}\t D_X:{:.2f}\tD_Y:{:.2f}\t label acc: {:.3f}|{:.3f}  batch:{} used: {:.2f} iter: {}'.format(
                   dict_losses['G_loss'], dict_losses['F_loss'], dict_losses['D_X_loss'], dict_losses['D_Y_loss'], X_groundtruth_acc, Y_groundtruth_acc,
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
