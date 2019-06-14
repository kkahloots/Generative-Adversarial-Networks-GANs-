"""
GAN_graph.py: Tensorflow Graph for the Generative Adversarial Networks
"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"

from base.base_graph import BaseGraph
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from networks.dense_net import DenseNet

'''
This is the Main GANGraph.
'''

class GANGraph(BaseGraph):
    def __init__(self, network_params, sigma_act=tf.nn.softplus,
                 transfer_fct=tf.nn.relu, learning_rate=1e-4,
                 kinit=tf.contrib.layers.xavier_initializer(), batch_size=32,
                 reuse=None, dropout=0.2):
        super().__init__(learning_rate)

        self.width = network_params['input_width']
        self.height = network_params['input_height']
        self.nchannel = network_params['input_nchannels']

        self.hidden_dim = network_params['hidden_dim']
        self.latent_dim = network_params.latent_dim
        self.num_layers = network_params['num_layers']  # Num of Layers in P(x|z)
        self.l2 = network_params.l2
        self.dropout = dropout

        self.sigma_act = sigma_act  # Actfunc for NN modeling variance

        self.x_flat_dim = self.width * self.height * self.nchannel

        self.transfer_fct = transfer_fct
        self.kinit = kinit
        self.bias_init = tf.constant_initializer(0.0)
        self.batch_size = batch_size

        self.reuse = reuse

    def build_graph(self):
        self.create_inputs()
        self.create_graph()
        self.create_loss_optimizer()

    def create_inputs(self):
        with tf.variable_scope('inputs', reuse=self.reuse):
            self.x_batch = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, self.nchannel], name='x_batch')
            self.x_batch_flat = tf.reshape(self.x_batch, [-1, self.x_flat_dim])
            self.z_batch = tf.placeholder(tf.float32, [self.batch_size, self.latent_dim], name='z_batch')

    ''' 
    ------------------------------------------------------------------------------
                                     GRAPH FUNCTIONS
    ------------------------------------------------------------------------------ 
    '''

    def create_graph(self):
        print('\n[*] Defining Generator...')
        with tf.variable_scope('generator', reuse=self.reuse):
            self.Gx = self.create_decoder(input_=self.z_batch,
                                          hidden_dim=self.hidden_dim,
                                          output_dim=self.x_flat_dim,
                                          num_layers=self.num_layers,
                                          transfer_fct=self.transfer_fct,
                                          act_out=tf.nn.sigmoid,
                                          reuse=self.reuse,
                                          kinit=self.kinit,
                                          bias_init=self.bias_init,
                                          drop_rate=self.dropout,
                                          prefix='g_'
                                          )

            self.x_recons = tf.reshape(self.Gx.output, [-1, self.width, self.height, self.nchannel])
            self.x_recons_flat = tf.reshape(self.Gx.output, [-1, self.x_flat_dim])
        print('\n[*] Defining Discriminator...')
        with tf.variable_scope('discriminator', reuse=self.reuse):
            self.D_real_logit = self.create_encoder(input_=self.x_batch_flat,
                                                    hidden_dim=self.hidden_dim,
                                                    output_dim=self.latent_dim,
                                                    num_layers=self.num_layers,
                                                    transfer_fct=self.transfer_fct,
                                                    act_out=None,
                                                    reuse=self.reuse,
                                                    kinit=self.kinit,
                                                    bias_init=self.bias_init,
                                                    drop_rate=self.dropout,
                                                    prefix='d_r_')

            self.D_real_prob = tf.nn.sigmoid(self.D_real_logit.output)

            self.D_fake_logit = self.create_encoder(input_=self.x_recons_flat,
                                                    hidden_dim=self.hidden_dim,
                                                    output_dim=self.latent_dim,
                                                    num_layers=self.num_layers,
                                                    transfer_fct=self.transfer_fct,
                                                    act_out=None,
                                                    reuse=self.reuse,
                                                    kinit=self.kinit,
                                                    bias_init=self.bias_init,
                                                    drop_rate=self.dropout,
                                                    prefix='d_f_')

            self.D_fake_prob = tf.nn.sigmoid(self.D_fake_logit.output)

    '''  
    ------------------------------------------------------------------------------
                                     ENCODER-DECODER
    ------------------------------------------------------------------------------ 
    '''

    def create_encoder(self, input_, hidden_dim, output_dim, num_layers, transfer_fct, \
                       act_out, reuse, kinit, bias_init, drop_rate, prefix):
        latent_ = DenseNet(input_=input_,
                           hidden_dim=hidden_dim,
                           output_dim=output_dim,
                           num_layers=num_layers,
                           transfer_fct=transfer_fct,
                           act_out=act_out,
                           reuse=reuse,
                           kinit=kinit,
                           bias_init=bias_init,
                           drop_rate=drop_rate,
                           prefix=prefix)
        return latent_

    def create_decoder(self, input_, hidden_dim, output_dim, num_layers, transfer_fct, \
                       act_out, reuse, kinit, bias_init, drop_rate, prefix):
        recons_ = DenseNet(input_=input_,
                           hidden_dim=hidden_dim,
                           output_dim=output_dim,
                           num_layers=num_layers,
                           transfer_fct=transfer_fct,
                           act_out=act_out,
                           reuse=reuse,
                           kinit=kinit,
                           bias_init=bias_init,
                           drop_rate=drop_rate,
                           prefix=prefix)
        return recons_

    '''  
    ------------------------------------------------------------------------------
                                     LOSSES
    ------------------------------------------------------------------------------ 
    '''

    def create_loss_optimizer(self):
        print('[*] Defining Loss Functions and Optimizer...')
        with tf.variable_scope('d_loss', reuse=self.reuse):
            # get loss for discriminator
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_real_logit.output, labels=tf.ones_like(self.D_real_prob)))

            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_fake_logit.output, labels=tf.zeros_like(self.D_fake_prob)))

            self.d_loss = self.d_loss_real + self.d_loss_fake

        with tf.variable_scope('g_loss', reuse=self.reuse):
            # get loss for generator
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_fake_logit.output, labels=tf.ones_like(self.D_fake_prob)))

        with tf.name_scope('reconstruct'):
            self.reconstruction = self.get_ell(self.x_batch_flat, self.x_recons_flat)

        t_vars = tf.trainable_variables()
        d_r_vars = [var for var in t_vars if 'd_r_' in var.name]
        d_f_vars = [var for var in t_vars if 'd_f_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        with tf.variable_scope("L2_loss", reuse=self.reuse):
            self.gL2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in g_vars])
            self.drL2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in d_r_vars])
            self.dfL2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in d_f_vars])
            self.reconL2_loss = self.gL2_loss + self.dfL2_loss

        self.g_loss = self.g_loss + self.l2 * self.gL2_loss
        self.d_loss = self.d_loss + self.l2 * self.dfL2_loss

        with tf.variable_scope("g_optimizer", reuse=self.reuse):
            self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=0.5)
            self.g_train_step = self.g_optimizer.minimize(self.g_loss, global_step=self.global_step_tensor)

        with tf.variable_scope("d_optimizer", reuse=self.reuse):
            self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
            self.d_train_step = self.d_optimizer.minimize(self.d_loss, global_step=self.global_step_tensor)

        with tf.variable_scope("ae_loss", reuse=self.reuse):
            self.ae_loss = tf.reduce_mean(self.reconstruction) + self.l2 * self.reconL2_loss  # shape = [None,]

        with tf.variable_scope("ae_optimizer", reuse=self.reuse):
            self.ae_optimizer = tf.train.AdamOptimizer(1.0)
            self.ae_train_step = self.ae_optimizer.minimize(self.ae_loss, global_step=self.global_step_tensor)

    ## ------------------- LOSS: EXPECTED LOWER BOUND ----------------------
    def get_ell(self, x, x_recons):
        """
        Returns the expected log-likelihood of the lower bound.
        For this we use a bernouilli LL.
        """
        # p(x|w)
        return - tf.reduce_sum((x) * tf.log(x_recons + 1e-10) +
                               (1 - x) * tf.log(1 - x_recons + 1e-10), 1)

    '''  
    ------------------------------------------------------------------------------
                                     FIT & EVALUATE TENSORS
    ------------------------------------------------------------------------------ 
    '''

    def partial_fit(self, session, x, z):
        tensors = [self.g_train_step, self.d_train_step, self.g_loss, self.gL2_loss, self.d_loss, self.d_loss_real,
                   self.d_loss_fake, self.dfL2_loss]
        feed_dict = {self.x_batch: x, self.z_batch: z}
        _, _, g_loss, gL2_loss, d_loss, d_loss_real, d_loss_fake, dL2_loss = session.run(tensors, feed_dict=feed_dict)
        return g_loss, gL2_loss, d_loss, d_loss_real, d_loss_fake, dL2_loss

    def evaluate(self, session, x, z):
        tensors = [self.g_loss, self.gL2_loss, self.d_loss, self.d_loss_real, self.d_loss_fake, self.dfL2_loss]
        feed_dict = {self.x_batch: x, self.z_batch: z}
        g_loss, gL2_loss, d_loss, d_loss_real, d_loss_fake, dL2_loss = session.run(tensors, feed_dict=feed_dict)
        return g_loss, gL2_loss, d_loss, d_loss_real, d_loss_fake, dL2_loss

    '''  
    ------------------------------------------------------------------------------
                                         GRAPH OPERATIONS
    ------------------------------------------------------------------------------ 
    '''

    def encode(self, session, inputs):
        tensors = [self.ae_train_step, self.ae_loss]
        feed_dict = {self.x_batch: inputs, self.z_batch: self.z_var}
        _, ae_loss = session.run(tensors, feed_dict=feed_dict)
        ae_loss = np.repeat(np.array([ae_loss]), inputs.shape[0]).reshape((-1, 1))

        return ae_loss

    def decode(self, session, z):
        tensors = [self.x_recons]
        feed_dict = {self.z_batch: z}
        return session.run(tensors, feed_dict=feed_dict)

