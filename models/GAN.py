"""GAN.py:  Autoencoder methods"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"


import os
import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')

from .GAN_BASE import GAN_BASE
import utils.utils as utils
import utils.constants as const
from sklearn.cluster import MiniBatchKMeans
import hdbscan

class GAN(GAN_BASE):
    def __init__(self, *argz, **kwrds):
        GAN_BASE.__init__(self, *argz, **kwrds)
        self.config.model_name = 'GAN'
        self.config.model_type = const.GAN
        self.setup_logging()

        
    def _build(self):
       
        '''  ---------------------------------------------------------------------
                            COMPUTATION GRAPH (Build the model)
        ---------------------------------------------------------------------- '''
        from Alg_GAN.GAN_model import GANModel
        self.model = GANModel(self.network_params, act_out=utils.softplus_bias,
                              transfer_fct=tf.nn.relu, learning_rate=self.config.l_rate,
                              kinit=tf.contrib.layers.xavier_initializer(),
                              batch_size=self.config.batch_size, dropout=self.config.dropout, batch_norm=self.config.batch_norm,
                              epochs=self.config.epochs, checkpoint_dir=self.config.checkpoint_dir,
                              summary_dir=self.config.summary_dir, result_dir=self.config.results_dir,
                              restore=self.flags.restore, plot=self.flags.plot, colab=self.flags.colab, model_type=self.config.model_type)
        print('building GAN Model...')
        print('\nNumber of trainable parameters', self.model.trainable_count)
        
    
    def animate(self):
        return self.model.animate()

    '''  
    ------------------------------------------------------------------------------
                                         MODEL OPERATIONS
    ------------------------------------------------------------------------------ 
    '''    
              
    def encode(self, inputs):
        '''  ------------------------------------------------------------------------------
                                         DATA PROCESSING
        ------------------------------------------------------------------------------ '''           
        inputs = utils.prepare_dataset(inputs) 
        return self.model.encode(inputs)
        
    def decode(self, w):
        return self.model.decode(w)
     
    def interpolate(self, input1, input2):
        input1 = utils.prepare_dataset(input1)
        input2 = utils.prepare_dataset(input2)         
        return self.model.interpolate(input1, input2)

    def reconst_loss(self, inputs):
        '''  ------------------------------------------------------------------------------
                                         DATA PROCESSING
        ------------------------------------------------------------------------------ '''           
        inputs = utils.prepare_dataset(inputs) 
        return self.model.reconst_loss(inputs)

    def do_clustering(self, x, alg='kmeans'):
        w = self.encode(x)
        if alg=='kmeans':
            clustering = MiniBatchKMeans(verbose=True)
            y_pred = clustering.fit_predict(w)
        else:
            clustering = hdbscan.HDBSCAN(min_cluster_size=50, gen_min_span_tree=False)
            clustering = clustering.fit(w)
            y_pred = clustering.labels_
        del clustering, w
        return y_pred