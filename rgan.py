import numpy as np
import tensorflow as tf

from .gan import GAN


class RGAN(GAN):
    def init_params(self):
        self.hidden_units_g = 150
        self.hidden_units_d = 150
        self.num_generated_features = 1
        self.latent_dim = 20

    def init_z_and_x(self):
        self.Z = tf.placeholder(tf.float32, [self.batch_size, self.seq_length, self.latent_dim])
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.seq_length, self.num_generated_features])

    def sample_Z(self):
        sample = np.float32(np.random.normal(size=[self.batch_size, self.seq_length, self.latent_dim]))
        return sample

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            generator_input = z
            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_units_g)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32,
                                                        sequence_length=[self.seq_length] * self.batch_size,
                                                        inputs=generator_input)
            rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, self.hidden_units_g])

            logits_2d = tf.layers.dense(rnn_outputs_2d,
                                        units=1,
                                        kernel_initializer=tf.initializers.truncated_normal,
                                        bias_initializer=tf.initializers.truncated_normal)
            output_2d = tf.nn.tanh(logits_2d)
            output_3d = tf.reshape(output_2d, [-1, self.seq_length, self.num_generated_features])
        return output_3d

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            decoder_input = x

            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_units_d)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=decoder_input)
            rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, self.hidden_units_g])
            logits = tf.layers.dense(rnn_outputs_flat,
                                     units=1,
                                     kernel_initializer=tf.initializers.truncated_normal,
                                     bias_initializer=tf.initializers.truncated_normal)
            #output = tf.nn.sigmoid(logits)

        return logits