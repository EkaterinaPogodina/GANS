import numpy as np
import tensorflow as tf

from .gan import GAN


class DCGAN(GAN):
    def init_z_and_x(self):
        self.Z = tf.placeholder(tf.float32, [self.batch_size, self.seq_length, 1])
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.seq_length, 1])

    def sample_Z(self):
        sample = np.float32(np.random.normal(size=[self.batch_size, self.seq_length, 1]))
        return sample

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            dense1 = tf.layers.conv1d(z, filters=128, kernel_size=2, padding='same', activation='relu')
            dense2 = tf.layers.conv1d(dense1, filters=64, kernel_size=4, padding='same', activation='relu')
            dense3 = tf.layers.conv1d(dense2, filters=32, kernel_size=4, padding='same', activation='relu')
            dense3 = tf.reshape(dense3, shape=(self.batch_size, 32 * self.seq_length))
            output = tf.layers.dense(dense3, units=self.seq_length)

        return tf.reshape(output, shape=(self.batch_size, self.seq_length, 1))

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse) as scope:
            if reuse:
                scope.reuse_variables()

            dense1 = tf.layers.conv1d(x, filters=128, kernel_size=2, padding='same', activation='relu')
            dense2 = tf.layers.conv1d(dense1, filters=64, kernel_size=4, padding='same', activation='relu')
            dense3 = tf.layers.conv1d(dense2, filters=32, kernel_size=4, padding='same', activation='relu')
            dense3 = tf.reshape(dense3, shape=(self.batch_size, 32 * self.seq_length))
            logits = tf.layers.dense(dense3, units=1)

        return logits

