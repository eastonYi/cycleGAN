import tensorflow as tf


def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


def random_crop(image, IMG_HEIGHT=256, IMG_WIDTH=256):
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


# normalizing the images from [0, 1] to [-1, 1]
def normalize(image):
    image = (image - 0.5) * 2
    return image


def denormalize(image):
    image = (image * 0.5) + 0.5
    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_image_train(image):
    image = load_image(image)
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image):
    image = load_image(image)
    image = normalize(image)
    return image
