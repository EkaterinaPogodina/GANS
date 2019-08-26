import tensorflow as tf
import numpy as np


def get_batch(samples, batch_size, idx):
    return samples[idx:idx + batch_size]


class GAN(object):
    def __init__(self, num_epochs=5, seq_length=10):

        tf.reset_default_graph()

        self.batch_size = 28
        self.seq_length = seq_length
        self.lr = 0.001
        self.num_epochs = num_epochs
        self.vis_freq = 2

        self.labels = None
        self.D_rounds = 3
        self.G_rounds = 1

        self.init_params()

        self.init_z_and_x()

        D_logit_real = self.get_D_logit_real()
        D_logit_fake = self.get_D_logit_fake()

        self.reg = self.get_reg()

        self.generator_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
        self.discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

        self.count_losses(D_logit_real, D_logit_fake)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def init_params(self):
        pass

    def get_reg(self):
        return 0

    def init_z_and_x(self):
        self.Z = tf.placeholder(tf.float32, [self.batch_size, self.seq_length])
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.seq_length])

    def get_D_logit_real(self):
        return self.discriminator(self.X)

    def get_D_logit_fake(self):
        self.G_sample = self.generator(self.Z)
        return self.discriminator(self.G_sample, reuse=True)

    def count_losses(self, D_logit_real, D_logit_fake):
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))

        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        self.D_solver = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.D_loss,
                                                                                     var_list=self.discriminator_vars)
        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.generator_vars)

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            dense = tf.layers.dense(z, units=128, activation='relu')
            output = tf.layers.dense(dense, units=10)
        return output

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse) as scope:
            if reuse:
                scope.reuse_variables()
            dense = tf.layers.dense(x, units=128, activation='relu')
            logits = tf.layers.dense(dense, units=1)

        return logits

    def sample_Z(self):
        sample = np.float32(np.random.normal(size=[self.batch_size, self.seq_length]))
        return sample

    def train_generator(self, batch_idx, offset, samples, reshape=False):
        for g in range(self.G_rounds):
            Y_mb = get_batch(samples, self.batch_size, batch_idx + g + offset)
            if reshape:
                Y_mb = Y_mb.reshape(self.batch_size, self.seq_length, 1)
            _, G_loss_curr = self.sess.run([self.G_solver, self.G_loss],
                                           feed_dict={self.Z: self.sample_Z(), self.X: Y_mb})

        return G_loss_curr

    def train_discriminator(self, batch_idx, offset, samples, reshape=False):

        for d in range(self.D_rounds):
            X_mb = get_batch(samples, self.batch_size, batch_idx + d + offset)
            if reshape:
                X_mb = X_mb.reshape(self.batch_size, self.seq_length, 1)

            _, D_loss_curr = self.sess.run([self.D_solver, self.D_loss],
                                      feed_dict={self.X: X_mb, self.Z: self.sample_Z()})
        return D_loss_curr

    def train_loop(self, samples, reshape=False):
        self.d_loss = []
        self.g_loss = []

        for num_epoch in range(self.num_epochs):

            for batch_idx in range(0, int(len(samples) / self.batch_size) - (self.D_rounds + self.G_rounds), self.D_rounds + self.G_rounds):
                if num_epoch % 2 == 0:

                    D_loss_curr = self.train_discriminator(batch_idx, self.G_rounds, samples, reshape)
                    G_loss_curr = self.train_generator(batch_idx, 0, samples, reshape)

                else:

                    D_loss_curr = self.train_discriminator(batch_idx, 0, samples, reshape)
                    G_loss_curr = self.train_generator(batch_idx, self.D_rounds, samples, reshape)

                self.d_loss.append(D_loss_curr)
                self.g_loss.append(G_loss_curr)

                if batch_idx % 1000 == 0:

                    print("epoch:", num_epoch, "\tD_loss:", D_loss_curr, "\tG_loss:", G_loss_curr, "\tTotal_loss:", D_loss_curr + G_loss_curr)

                    gen_samples = []
                    for batch_idx in range(int(len(samples) / self.batch_size)):
                        gen_samples_mb = self.sess.run(self.G_sample, feed_dict={self.Z: self.sample_Z()})
                        gen_samples.append(gen_samples_mb)

                gen_samples = np.vstack(gen_samples)
        return gen_samples

    def reset(self):
        self.__init__()
