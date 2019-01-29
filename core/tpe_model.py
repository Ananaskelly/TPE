import tensorflow as tf


class TPEModel:

    def __init__(self, emb_size, hid_size):
        self.emb_size = emb_size
        self.hid_size = hid_size

        self.anchor = None
        self.positive = None
        self.negative = None

        self.score = None
        self.loss = None
        self.optimize = None

        self.optimizer = tf.train.AdamOptimizer(0.001)

    def build_network(self):

        self.anchor = tf.placeholder([-1, self.emb_size])
        self.positive = tf.placeholder([-1, self.emb_size])
        self.negative = tf.placeholder([-1, self.emb_size])

        self.score = self.network()

        self.loss = self.triplet_loss()
        self.optimize = self.optimize()

    def network(self):

        anchor_emb = tf.layers.dense(self.anchor, units=self.hid_size, use_bias=False)
        positive_emb = tf.layers.dense(self.positive, units=self.hid_size, use_bias=False)
        negative_emb = tf.layers.dense(self.negative, units=self.hid_size, use_bias=False)

        score = tf.math.multiply(anchor_emb, positive_emb) - tf.math.multiply(anchor_emb, negative_emb)

        return score

    def triplet_loss(self):

        return tf.math.reduce_mean(tf.math.log(tf.math.sigmoid(self.score)))

    def minimize(self):

        return self.optimizer.minimize(self.loss)
