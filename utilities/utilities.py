import tensorflow as tf

def normalize(img):
    normalized = (img / 127.5) - 1
    return normalized

def AdaIN2D(content, style):
    content_mean, content_var = tf.nn.moments(content, axes=[1, 2])
    style_mean, style_var = tf.nn.moments(style, axes=[1, 2])
    content_std = tf.sqrt(content_var)
    style_std = tf.sqrt(style_var)

    content_mean = tf.broadcast_to(content_mean, tf.shape(content))
    content_std = tf.broadcast_to(content_std, tf.shape(content))
    style_mean = tf.broadcast_to(style_mean, tf.shape(content))
    style_std = tf.broadcast_to(style_std, tf.shape(content))

    normalized_content = tf.divide(content - content_mean, content_std)

    return tf.multiply(normalized_content, style_std) + style_mean
