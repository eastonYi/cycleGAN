import tensorflow as tf
from models import pix2pix
from utils.arguments import args
from utils.img_tools import preprocess_image_train, preprocess_image_test, random_jitter, denormalize
from time import time
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE
np.set_printoptions(precision=3)
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 10
EPOCHS = 40


def main(args):
    """The basic CycleGAN template.

    Args:
        features: a dict containing key "X" and key "Y"
        mode: training, evaluation or infer
    """
    # load dataset
    PATH = '/home/user/easton/data/horse2zebra/'

    train_horses = tf.data.Dataset.list_files(PATH + 'trainA/*.jpg', shuffle=False).map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    train_zebras = tf.data.Dataset.list_files(PATH + 'trainB/*.jpg', shuffle=False).map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_horses = tf.data.Dataset.list_files(PATH + 'testA/*.jpg', shuffle=False).map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_zebras = tf.data.Dataset.list_files(PATH + 'testB/*.jpg', shuffle=False).map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    sample_horse = next(iter(train_horses))
    sample_zebra = next(iter(train_zebras))

    plt.subplot(121)
    plt.title('Horse')
    plt.imshow(denormalize(sample_horse[0]))

    plt.subplot(122)
    plt.title('Horse with random jitter')
    plt.imshow(denormalize(random_jitter(sample_horse[0])))

    plt.savefig('peek.png')

    # gather transformations and descriminators
    OUTPUT_CHANNELS = 3

    G = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    F = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

    D_X = pix2pix.discriminator(norm_type='instancenorm', target=False)
    D_Y = pix2pix.discriminator(norm_type='instancenorm', target=False)

    to_zebra = G(sample_horse)
    to_horse = F(sample_zebra)
    plt.figure(figsize=(8, 8))
    contrast = 8 # to see it carefully since init model output is poor
    imgs = [sample_horse, to_zebra, sample_zebra, to_horse]
    title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

    for i in range(len(imgs)):
        plt.subplot(2, 2, i+1)
        plt.title(title[i])
        if i % 2 == 0:
            plt.imshow(denormalize(imgs[i][0]))
        else:
            plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
    plt.savefig('init.png')

    plt.figure(figsize=(8, 8))

    plt.subplot(121)
    plt.title('Is a real zebra?')
    plt.imshow(D_Y(sample_zebra)[0, ..., -1], cmap='RdBu_r')

    plt.subplot(122)
    plt.title('Is a real horse?')
    plt.imshow(D_X(sample_horse)[0, ..., -1], cmap='RdBu_r')

    plt.savefig('init.png')

    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    @tf.function
    def train_step(real_x, real_y):
      # persistent is set to True because gen_tape and disc_tape is used more than
      # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.
            fake_y = G(real_x, training=True)
            cycled_x = F(fake_y, training=True)

            fake_x = F(real_y, training=True)
            cycled_y = G(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = F(real_x, training=True)
            same_y = G(real_y, training=True)

            disc_real_x = D_X(real_x, training=True)
            disc_real_y = D_Y(real_y, training=True)

            disc_fake_x = D_X(fake_x, training=True)
            disc_fake_y = D_Y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = generator_loss(disc_fake_y, loss_obj)
            gen_f_loss = generator_loss(disc_fake_x, loss_obj)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + calc_cycle_loss(real_x, cycled_x) + identity_loss(real_x, same_x)
            total_gen_f_loss = gen_f_loss + calc_cycle_loss(real_y, cycled_y) + identity_loss(real_y, same_y)

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x, loss_obj)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y, loss_obj)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = gen_tape.gradient(total_gen_g_loss, G.trainable_variables)
        generator_f_gradients = gen_tape.gradient(total_gen_f_loss, F.trainable_variables)
        discriminator_x_gradients = disc_tape.gradient(disc_x_loss, D_X.trainable_variables)
        discriminator_y_gradients = disc_tape.gradient(disc_y_loss, D_Y.trainable_variables)
        del gen_tape
        del disc_tape

        # Apply the gradients to the optimizer
        generator_g_optimizer.apply_gradients(
            zip(generator_g_gradients,
                G.trainable_variables))

        generator_f_optimizer.apply_gradients(
            zip(generator_f_gradients,
                F.trainable_variables))

        discriminator_x_optimizer.apply_gradients(
            zip(discriminator_x_gradients,
                D_X.trainable_variables))

        discriminator_y_optimizer.apply_gradients(
            zip(discriminator_y_gradients,
                D_Y.trainable_variables))

        return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss

    for epoch in range(EPOCHS):
        start = time()

        for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
            total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss = train_step(image_x, image_y)
            print('total_gen_g_loss: {:.3f}, \ttotal_gen_f_loss: {:.3f}, \tdisc_x_loss: {:.3f}, \tdisc_y_loss: {:.3f}'.format(
                total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss))
        # Using a consistent image (sample_horse) so that the progress of the model
        # is clearly visible.
        generate_images(G, sample_horse, name='dev-'+str(epoch))
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time()-start))

    test(test_horses, G)



def discriminator_loss(real, generated, loss_obj):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5

def generator_loss(generated, loss_obj):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


def generate_images(model, test_input, name):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(denormalize(display_list[i]))
        plt.axis('off')
    plt.savefig(name + '.png')


def test(test_horses, G):
    # Run the trained model on the test dataset
    for inp in test_horses.take(5):
        generate_images(G, inp, name='test')


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