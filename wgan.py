import tensorflow as tf
import tensorflow.contrib as tc

from .dcgan import DCGAN
from .gan import get_batch


class WGAN(DCGAN):
    def get_reg(self):
        weights_list = [var for var in tf.global_variables()]
        return tc.layers.apply_regularization(
            tc.layers.l1_regularizer(2.5e-5),
            weights_list
        )

    # def get_D_logit_real(self):
    #     return tf.reduce_mean(self.discriminator(self.X))
    #
    # def get_D_logit_fake(self):
    #     self.G_sample = self.generator(self.Z)
    #     return tf.reduce_mean(self.discriminator(self.G_sample, reuse=True))

    def count_losses(self, D_logit_real, D_logit_fake):
        # self.D_loss = D_logit_real + self.reg
        # self.G_loss = D_logit_fake + self.reg

        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))

        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        self.D_loss += self.reg
        self.G_loss += self.reg


        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5) \
                .minimize(self.D_loss, var_list=self.discriminator_vars)
            self.G_solver = tf.train.RMSPropOptimizer(learning_rate=5e-5) \
                .minimize(self.G_loss, var_list=self.generator_vars)

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.discriminator_vars]

    def train_discriminator(self, batch_idx, offset, samples, reshape=False):

        for d in range(self.D_rounds):
            X_mb = get_batch(samples, self.batch_size, batch_idx + d + offset)
            if reshape:
                X_mb = X_mb.reshape(self.batch_size, self.seq_length, 1)

            self.sess.run(self.d_clip)
            _, D_loss_curr = self.sess.run([self.D_solver, self.D_loss],
                                      feed_dict={self.X: X_mb, self.Z: self.sample_Z()})
        return D_loss_curr