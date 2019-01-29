import tensorflow as tf


def weight_variable(shape, name='', w=0.1):
    initial = tf.truncated_normal(shape, stddev=w)

    if name != '':
        return tf.Variable(initial, name=name)

    return tf.Variable(initial)


def bias_variable(shape, name='', w =0.001):
    initial = tf.truncated_normal(shape, stddev=w)

    if name != '':
        return tf.Variable(initial, name=name)

    return tf.Variable(initial)


def weight_variable_xavier(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))
