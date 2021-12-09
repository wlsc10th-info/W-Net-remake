import tensorflow as tf

discriminator_dim = 32

# Helper function to return a tf Conv2D layer with specified filters
def conv2d(filters, kernel_size=(5, 5), strides=(2, 2), padding='same'):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)

def lrelu():
    return tf.keras.layers.LeakyReLU(alpha=0.2)

def layer_norm():
    return tf.keras.layers.LayerNormalization()

class Discriminator(tf.keras.Model):

    def __init__(self, output_size):
        super().__init__()

        self.conv2d_1 = conv2d(discriminator_dim)
        self.conv2d_2 = conv2d(discriminator_dim * 2)
        self.conv2d_3 = conv2d(discriminator_dim * 4)
        self.conv2d_4 = conv2d(discriminator_dim * 8)
        self.conv2d_5 = conv2d(discriminator_dim * 16)
        self.conv2d_6 = conv2d(discriminator_dim * 32)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_size)

    def call(self, x):
        x = lrelu()(self.conv2d_1(x))
        x = lrelu()(layer_norm()(self.conv2d_2(x)))
        x = lrelu()(layer_norm()(self.conv2d_3(x)))
        x = lrelu()(layer_norm()(self.conv2d_4(x)))
        x = lrelu()(layer_norm()(self.conv2d_5(x)))
        x = lrelu()(layer_norm()(self.conv2d_6(x)))
        x = self.flatten(x)
        x = self.dense(x)
        return x
