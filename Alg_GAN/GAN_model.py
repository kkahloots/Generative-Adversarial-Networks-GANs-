"""GAN_model.py: Tensorflow model for the Generative Adversarial Networks"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"

import sys

sys.path.append('..')
import gc
import glob

from PIL import Image as PILImage
from IPython.display import display, Image
import os

from base.base_model import BaseModel
import tensorflow as tf
import numpy as np

from .GAN_graph import GANGraph
from .GANCNN_graph import GANCNNGraph

from utils.logger import Logger
from utils.early_stopping import EarlyStopping
from tqdm import tqdm
import sys
from collections import defaultdict

import utils.utils as utils
import utils.constants as const

from sklearn.decomposition import PCA
from utils.plots import plot_dataset, plot_samples, merge, resize_gif


class GANModel(BaseModel):
    def __init__(self, network_params, act_out=tf.nn.softplus,
                 transfer_fct=tf.nn.relu, learning_rate=0.002,
                 kinit=tf.contrib.layers.xavier_initializer(), batch_size=32,
                 latent_dim=10, dropout=0.2, batch_norm=True, epochs=200, checkpoint_dir='',
                 summary_dir='', result_dir='', restore=False, plot=False, colab=False, model_type=const.GAN):
        super().__init__(checkpoint_dir, summary_dir, result_dir)

        self.summary_dir = summary_dir
        self.result_dir = result_dir
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.epochs = epochs
        self.z_file = result_dir + '/z_file'
        self.restore = restore
        self.colab = colab
        self.plot = plot

        if self.plot:
            self.z_space_files = list()
            self.recons_files = list()

        # Creating computational graph for train and test
        self.graph = tf.Graph()
        with self.graph.as_default():
            if (model_type == const.GAN):
                self.model_graph = GANGraph(network_params, act_out,
                                            transfer_fct, learning_rate, kinit, batch_size,
                                            reuse=False)
            if (model_type == const.GANCNN):
                self.model_graph = GANCNNGraph(network_params, act_out,
                                               transfer_fct, learning_rate, kinit, batch_size,
                                               reuse=False)

            self.model_graph.build_graph()
            self.trainable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    def train_epoch(self, session, logger, data_train):
        loop = tqdm(range(data_train.num_batches(self.batch_size)))
        d_loss = []
        d_loss_real = []
        d_loss_fake = []
        dL2_loss = []

        g_loss = []
        gL2_loss = []

        for _ in loop:
            batch_x = next(data_train.next_batch(self.batch_size))
            self.batch_z = np.random.uniform(-1, 1, [self.batch_size, self.latent_dim]).astype(np.float32)
            g_loss_curr, gL2_loss_curr, d_loss_curr, d_loss_real_curr, d_loss_fake_curr, dL2_loss_curr = self.model_graph.partial_fit(
                session, batch_x, self.batch_z)

            g_loss.append(g_loss_curr)
            gL2_loss.append(gL2_loss_curr)

            d_loss.append(d_loss_curr)
            d_loss_real.append(d_loss_real_curr)
            d_loss_fake.append(d_loss_fake_curr)
            dL2_loss.append(dL2_loss_curr)

        g_loss_tr = np.mean(g_loss)
        gL2_loss_tr = np.mean(gL2_loss)

        d_loss_tr = np.mean(d_loss)
        d_loss_real_tr = np.mean(d_loss_real)
        d_loss_fake_tr = np.mean(d_loss_fake)
        dL2_loss_tr = np.mean(dL2_loss)

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = {
            'g_loss_tr': g_loss_tr,
            'gL2_loss_tr': gL2_loss_tr,
            'd_loss_tr': d_loss_tr,
            'd_loss_real_tr': d_loss_real_tr,
            'd_loss_fake_tr': d_loss_fake_tr,
            'dL2_loss_tr': dL2_loss_tr
        }

        logger.summarize(cur_it, summaries_dict=summaries_dict)

        return g_loss_tr, gL2_loss_tr, d_loss_tr, d_loss_real_tr, d_loss_fake_tr, dL2_loss_tr

    def valid_epoch(self, session, logger, data_valid):
        loop = tqdm(range(data_valid.num_batches(self.batch_size)))
        d_loss = []
        d_loss_real = []
        d_loss_fake = []
        dL2_loss = []

        g_loss = []
        gL2_loss = []

        for _ in loop:
            batch_x = next(data_valid.next_batch(self.batch_size))
            g_loss_curr, gL2_loss_curr, d_loss_curr, d_loss_real_curr, d_loss_fake_curr, dL2_loss_curr = self.model_graph.evaluate(
                session, batch_x, self.batch_z)

            g_loss.append(g_loss_curr)
            gL2_loss.append(gL2_loss_curr)

            d_loss.append(d_loss_curr)
            d_loss_real.append(d_loss_real_curr)
            d_loss_fake.append(d_loss_fake_curr)
            dL2_loss.append(dL2_loss_curr)

        g_loss_val = np.mean(g_loss)
        gL2_loss_val = np.mean(gL2_loss)

        d_loss_val = np.mean(d_loss)
        d_loss_real_val = np.mean(d_loss_real)
        d_loss_fake_val = np.mean(d_loss_fake)
        dL2_loss_val = np.mean(dL2_loss)

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = {
            'g_loss_val': g_loss_val,
            'gL2_loss_val': gL2_loss_val,
            'd_loss_val': d_loss_val,
            'd_loss_real_val': d_loss_real_val,
            'd_loss_fake_val': d_loss_fake_val,
            'dL2_loss_val': dL2_loss_val
        }

        logger.summarize(cur_it, summaries_dict=summaries_dict)

        return g_loss_val, gL2_loss_val, d_loss_val, d_loss_real_val, d_loss_fake_val, dL2_loss_val

    def train(self, data_train, data_valid, enable_es=1):

        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(1234)

            logger = Logger(session, self.summary_dir)
            # here you initialize the tensorflow saver that will be used in saving the checkpoints.
            # max_to_keep: defaults to keeping the 5 most recent checkpoints of your model
            saver = tf.train.Saver()
            early_stopping = EarlyStopping(name='total loss')

            if (self.restore and self.load(session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                print('Initializing Variables ...')
                tf.global_variables_initializer().run()

            if (self.model_graph.cur_epoch_tensor.eval(session) == self.epochs):
                return

            for cur_epoch in range(self.model_graph.cur_epoch_tensor.eval(session), self.epochs + 1, 1):

                print('EPOCH: ', cur_epoch)
                self.current_epoch = cur_epoch

                g_loss_tr, gL2_loss_tr, d_loss_tr, d_loss_real_tr, d_loss_fake_tr, dL2_loss_tr = self.train_epoch(
                    session, logger, data_train)

                g_loss_val, gL2_loss_val, d_loss_val, d_loss_real_val, d_loss_fake_val, dL2_loss_val = self.valid_epoch(
                    session, logger, data_valid)

                print('TRAIN | GAN Generator: g_loss: ', g_loss_tr, ' | L2_loss: ', gL2_loss_tr)
                print('VALID | GAN Generator: g_loss: ', g_loss_val, ' | L2_loss: ', gL2_loss_val)
                print()
                print('TRAIN | GAN Discriminator: d_loss: ', d_loss_tr, ' | d_loss_real: ', d_loss_real_tr)
                print('      | d_loss_fake: ', d_loss_fake_tr, ' | L2_loss: ', dL2_loss_tr)
                print('VALID | GAN Discriminator: d_loss: ', d_loss_val, ' | d_loss_real: ', d_loss_real_val)
                print('      | d_loss_fake: ', d_loss_fake_val, ' | L2_loss: ', dL2_loss_val)

                if (cur_epoch == 1) or ((cur_epoch % const.SAVE_EPOCH == 0) and ((cur_epoch != 0))):
                    self.save(session, saver, self.model_graph.global_step_tensor.eval(session))
                    if self.plot:
                        self.generate_samples(data_train, session, cur_epoch)

                if cur_epoch % 50 == 0:
                    if self.colab:
                        self.push_colab()

                session.run(self.model_graph.increment_cur_epoch_tensor)

                # Early stopping
                # if(enable_es==1 and early_stopping.stop(g_loss_val)):
                # print('Early Stopping!')
                # break

            self.save(session, saver, self.model_graph.global_step_tensor.eval(session))
            if self.plot:
                self.generate_samples(data_train, session, cur_epoch)

            if self.colab:
                self.push_colab()

        return

    '''  
    ------------------------------------------------------------------------------
                                         MODEL OPERATIONS
    ------------------------------------------------------------------------------ 
    '''

    def reconst(self, inputs):
        return self.decode(self.encode(inputs))

    def encode(self, inputs):
        self.model_graph.z_var = np.random.uniform(-1, 1, [self.batch_size, self.latent_dim]).astype(np.float32)
        for _ in range(10):
            self.model_graph.z_var = np.random.uniform(-1, 1, [self.batch_size, self.latent_dim]).astype(np.float32)
            ae_loss = self.batch_function(self.model_graph.encode, inputs)
            print('AE loss: {}'.format(np.mean(ae_loss)))
        z_var = np.repeat(self.model_graph.z_var, inputs.shape[0] // self.batch_size).reshape((-1, self.latent_dim))
        return z_var[:inputs.shape[0]]

    def decode(self, z):
        return self.batch_function(self.model_graph.decode, z)

    def interpolate(self, input1, input2):

        z1 = self.encode(input1)
        z2 = self.encode(input2)

        decodes = defaultdict(list)
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            decode = dict()
            z = np.stack([self.slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.decode(z)

            for i in range(z_decode.shape[0]):
                try:
                    decode[i] = [z_decode[i].compute()]
                except:
                    decode[i] = [z_decode[i]]

            for i in range(z_decode.shape[0]):
                decodes[i] = decodes[i] + decode[i]

        imgs = []

        for idx in decodes:
            l = []

            l += [input1[idx:idx + 1][0]]
            l += decodes[idx]
            l += [input2[idx:idx + 1][0]]

            imgs.append(l)
        del decodes

        return imgs

    def reconst_loss(self, inputs):
        return self.batch_function(self.model_graph.reconst_loss, inputs)

    def slerp(self, val, low, high):
        """Code from https://github.com/soumith/dcgan.torch/issues/14"""
        omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high.transpose() / np.linalg.norm(high)), -1, 1))
        so = np.sin(omega)

        # l1 = lambda low, high, val: (1.0-val) * low + val * high
        # l2 = lambda low, high, val, so, omega: np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
        if so == 0:
            return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
        return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

    '''
    ------------------------------------------------------------------------------
                                         GENERATE SAMPLES
    ------------------------------------------------------------------------------
    '''

    def animate(self):
        if not hasattr(self, 'z_space_files') or len(self.recons_files) == 0:
            print('No images were generated during trainning!')

            path = self.summary_dir
            st = path + '/{} samples generation in epoch'.format(self.summary_dir.split('/')[-1:][0])
            self.recons_files = [f for f in glob.glob(path + "**/*.jpg", recursive=True) if f.startswith(st)]
            self.recons_files = list(map(lambda f: f.split('/')[-1], self.recons_files))

            self.recons_files.sort(key=utils.natural_keys)
            self.recons_files = list(map(lambda f: path + '/' + f, self.recons_files))

            st = path + '/{} Z space in epoch'.format(self.summary_dir.split('/')[-1:][0])
            self.z_space_files = [f for f in glob.glob(path + "**/*.jpg", recursive=True) if f.startswith(st)]
            self.z_space_files = list(map(lambda f: f.split('/')[-1], self.z_space_files))

            self.z_space_files.sort(key=utils.natural_keys)
            self.z_space_files = list(map(lambda f: path + '/' + f, self.z_space_files))

            if len(self.recons_files) == 0:
                print('No previous images found!')
                return None, None

        path = self.summary_dir
        st = path + '/{} samples generation in epoch'.format(self.summary_dir.split('/')[-1:][0])
        images = [PILImage.open(fn) for fn in self.recons_files]

        images[0].save(st + '_animate.gif', save_all=True, append_images=images[1:], duration=len(images) * 60,
                       loop=0xffff)
        images[0].save(st + '_res_animate.gif', save_all=True, append_images=images[1:], duration=len(images) * 60,
                       loop=0xffff, dpi=70)
        with open(st + '_res_animate.gif', 'rb') as f:
            img1 = Image(data=f.read(), format='gif')

        st = path + '/{} Z space in epoch'.format(self.summary_dir.split('/')[-1:][0])
        images = [PILImage.open(fn) for fn in self.z_space_files]
        images[0].save(st + '_animate.gif', save_all=True, append_images=images[1:], duration=len(images) * 60,
                       loop=0xffff, dpi=70)

        resize_gif(path=st + '_animate.gif', save_as=st + '_res_animate.gif', resize_to=(900, 450))
        with open(st + '_res_animate.gif', 'rb') as f:
            img2 = Image(data=f.read(), format='gif')

        return img1, img2

    def generate_samples(self, data, session, cur_epoch=''):
        # Generating Z Space
        print('Generating Z Space ...')
        z_recons_l = self.encode(data.x)
        np.savez(self.z_file, z_recons_l)

        pca = PCA(n_components=2)
        Z_pca = pca.fit_transform(z_recons_l)
        print('Z space dimensions: {}'.format(Z_pca.shape))
        print('Ploting Z space ...')
        z_space = self.summary_dir + '/{} Z space in epoch {}.jpg'.format(self.summary_dir.split('/')[-1:][0],
                                                                           cur_epoch)
        self.z_space_files.append(z_space)
        plot_dataset(Z_pca, y=data.labels, save=z_space)

        del Z_pca, z_recons_l
        gc.collect()

        # Generating Samples
        print('Generating Samples ...')

        x_recons_l = self.reconst(data.samples)
        recons_file = self.summary_dir + '/{} samples generation in epoch {}.jpg'.format(
            self.summary_dir.split('/')[-1:][0], cur_epoch)
        self.recons_files.append(recons_file)
        plot_samples(x_recons_l, scale=10, save=recons_file)

        del x_recons_l
        gc.collect()

    ''' 
     ------------------------------------------------------------------------------
                                          MODEL FUNCTIONS
     ------------------------------------------------------------------------------ 
     '''

    def batch_function(self, func, p1):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            if (self.load(session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                return

            output_l = list()

            start = 0
            end = self.batch_size

            with tqdm(range(p1.shape[0] // self.batch_size)) as pbar:
                while end < p1.shape[0]:
                    output = func(session, p1[start:end])
                    output = np.array(output)
                    output = output.reshape([output.shape[0] * output.shape[1]] + list(output.shape[2:]))
                    output_l.append(output)

                    start = end
                    end += self.batch_size
                    pbar.update(1)
                else:

                    x1 = p1[start:]
                    xsize = len(x1)
                    p1t = np.zeros([self.batch_size - xsize] + list(x1.shape[1:]))

                    output = func(session, np.concatenate((x1, p1t), axis=0))
                    output = np.array(output)
                    output = output.reshape([output.shape[0] * output.shape[1]] + list(output.shape[2:]))[0:xsize]

                    output_l.append(output)

                    pbar.update(1)

        try:
            return np.vstack(output_l)
        except:
            output_l = list(map(lambda l: l.reshape(-1, 1), output_l))
        return np.vstack(output_l)
