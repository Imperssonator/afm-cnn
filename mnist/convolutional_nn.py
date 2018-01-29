# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:43:38 2017

@author: leiming.wang
"""
import numpy as np
import tensorflow as tf


def _variable_on_cpu(name, shape, initializer, trainable=True):
    """Helper to create a variable stored on CPU mem."""
    
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape,
                              initializer=initializer, dtype=tf.float32,
                              trainable=trainable)
    
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized variable with weight decay
    (i.e., L2 regularization)
    The variable is initialized with a truncated normal distribution.
    """
    
    var = _variable_on_cpu(
            name, shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
            
    if wd is not None:
        l2_loss = tf.multiply(tf.nn.l2_loss(var), wd, name='l2_loss')
        tf.add_to_collection('losses', l2_loss)
        
    return var
    


class ConvLayer:
    
    def __init__(self,
                 layer_name,
                 input_shape,
                 filter_shape,
                 strides=[1,1,1,1],
                 padding='SAME',
                 weight_decay=0.0):
        
        self.layer_name = layer_name
        self.input_shape = input_shape 
        self.filter_shape = filter_shape
        self.strides = strides
        self.padding = padding
        self.wd = weight_decay
          
        
        """
        #input format consistent with tensorflow tf.nn.conv2d default#
        input_shape -- [-1, in_height, in_width, in_channels]
        #note: use -1 to represent batch_size in input_shape for initialization
        filter_shape -- [filter_height, filter_width, im_channels, num_filters]
        strides -- [1, stride_h, stride_w, 1]
        padding -- string 'SAME', or 'VALID'
        weight_decay -- lambda for l2 regularization
        """
         
        #initialize weights and biases
        with tf.variable_scope(self.layer_name) as self.scope:
            n_in = self.filter_shape[0]*self.filter_shape[1]*self.filter_shape[2]
            n_out = self.filter_shape[0]*self.filter_shape[1]*self.filter_shape[-1]
            stddev = np.sqrt(2.0/(n_in + n_out))
              
            self.w = _variable_with_weight_decay(
                    'weights',
                    shape=self.filter_shape,
                    stddev=stddev,
                    wd=self.wd)
            
            self.b = _variable_on_cpu('biases', self.filter_shape[-1],
                                      tf.constant_initializer(0.0))
        
    
    def set_output(self, inpt, is_training):
        """
        set output of the layer with given input tensor
        """
        inpt = tf.reshape(inpt, self.input_shape)
        with tf.variable_scope(self.scope.original_name_scope):
            inpt = tf.reshape(inpt, shape=self.input_shape)       
            conv = tf.nn.conv2d(inpt,
                                self.w,
                                strides=self.strides,
                                padding=self.padding)
            
            self.output = tf.nn.bias_add(conv, self.b,
                                         name=self.scope.name)
            

class BatchNormLayer:
    
    def __init__(self,
                 layer_name,
                 input_shape,
                 decay=0.999,
                 epsilon=1.0e-5):
        
        self.layer_name = layer_name
        self.input_shape = input_shape
        self.decay = decay
        self.epsilon = epsilon
            
        """
        # initialize parameters for batch normalization:
        # sclale -- gamma
        # offset -- beta
        # pop_mean, pop_var record the moving average of batch_mean
        # and batch_variance, to be used duing testing after trained.
        """
        
        with tf.variable_scope(self.layer_name) as self.scope:
            self.gamma = _variable_on_cpu('scale', self.input_shape[-1],
                                     tf.constant_initializer(1.0))
            self.beta = _variable_on_cpu('offset', self.input_shape[-1],
                                    tf.constant_initializer(0.0))
            
            self.pop_mean = _variable_on_cpu('pop_mean', self.input_shape[-1],
                                             tf.constant_initializer(0.0),
                                             trainable=False)
            self.pop_var = _variable_on_cpu('pop_variance',
                                            self.input_shape[-1],
                                            tf.constant_initializer(1.0),
                                            trainable=False)
            
    
    def set_output(self, inpt, is_training):
        
        inpt = tf.reshape(inpt, self.input_shape)
        with tf.variable_scope(self.scope.original_name_scope):
            
            
            batch_mean, batch_var = \
                tf.nn.moments(inpt, axes=list(range(len(self.input_shape)-1)))
                        
            # re-assign mean and var with its exponetial moving averages    
            mean_ema_op = tf.assign(self.pop_mean,
                                    self.pop_mean * self.decay
                                    + batch_mean * (1.0-self.decay))
            
            var_ema_op = tf.assign(self.pop_var,
                                   self.pop_var * self.decay
                                   + batch_var * (1.0-self.decay))
            
            def mean_var_with_update():   
                with tf.control_dependencies([mean_ema_op, var_ema_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
                    # tf.identity fn is used as an operator under the
                    # control_dependencies to trigger the mean/var_op updates
                                                                         
            def mean_var_ave():
                return self.pop_mean, self.pop_var
            
            mean, var = tf.cond(is_training,
                                mean_var_with_update, mean_var_ave)
                                
                
            self.output = tf.nn.batch_normalization(inpt,
                                                    mean,
                                                    var,
                                                    self.beta,
                                                    self.gamma,
                                                    self.epsilon)
   



class FullyConnectedLayer:
    
    def __init__(self, layer_name, input_shape, output_shape, weight_decay=0.0):
        
        """
        # use -1 for batch_size shape
        input_shape = [-1, n_in]
        output_shape = [-1, n_out]
        """
        self.layer_name = layer_name
        self.n_in = input_shape[1] 
        self.n_out = output_shape[1]
        self.wd = weight_decay
        
        # initialize weights and biases
        with tf.variable_scope(layer_name) as self.scope:
            stddev = np.sqrt(2.0/(self.n_in + self.n_out))
            self.w = _variable_with_weight_decay(
                    'weights',
                    shape = [self.n_in, self.n_out],
                    stddev = stddev,
                    wd = self.wd)
            
            self.b = _variable_on_cpu('biases',
                                      [self.n_out],
                                      tf.constant_initializer(0.0))
                  
            
    def set_output(self, inpt, is_training):
        
        inpt = tf.reshape(inpt, [-1, self.n_in])
        with tf.variable_scope(self.scope.original_name_scope):
            self.output = tf.nn.bias_add(tf.matmul(inpt, self.w),
                                         self.b, name=self.scope.name)
    



class ReluLayer:
    
    def __init__(self, layer_name):
        self.layer_name = layer_name
        
        
    def set_output(self, inpt, is_training):      
        
        with tf.variable_scope(self.layer_name) as self.scope:
            self.output = tf.nn.relu(inpt, name=self.scope.name)



class SigmoidLayer:
    
    def __init__(self, layer_name):
        self.layer_name = layer_name
        
        
    def set_output(self, inpt, is_training):      
        
        with tf.variable_scope(self.layer_name) as self.scope:
            self.output = tf.nn.sigmoid(inpt, name=self.scope.name)




class MaxPoolLayer:
    
    def __init__(self, layer_name, ksize,
                 strides=[1,1,1,1],
                 padding='SAME'):

        self.layer_name = layer_name
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
                    
    def set_output(self, inpt, is_training):
               
        with tf.variable_scope(self.layer_name) as self.scope:
            self.output = tf.nn.max_pool(inpt,
                                         ksize=self.ksize,
                                         strides = self.strides,
                                         padding = self.padding,
                                         name = self.scope.name)    
             


class DropOutLayer:
    
    def __init__(self, layer_name, keep_prob):
        
        self.layer_name = layer_name
        self.keep_prob = keep_prob
    
    def set_output(self, inpt, is_training):
        
        with tf.variable_scope(self.layer_name) as self.scope:
            
            
            def train_phase():
                return tf.nn.dropout(inpt, self.keep_prob,
                                     name=self.scope.name)
                
            def test_phase():
                return tf.identity(inpt, name=self.scope.name)
            
            self.output = tf.cond(is_training, train_phase, test_phase)
        
        
        

###### Main class ######   
class Network:
    
    def __init__(self, layers):
        
        """
        Takes a list of "layers" to construct the computation graph
        """
        self.layers = layers

        
        
    def build_graph(self, images, labels, is_training):
        
        
        self.logits = self.inference(images, is_training)
            
        self.prediction, self.accuracy, self.accuracy_summary = \
                                    self.evaluation(self.logits, labels)
            
        self.loss, self.loss_summary = self.cost(self.logits, labels)
            
        
    
    def inference(self, x, is_training):
        """
        build the network graph for forward calculation from x to y.
        """
        # set output for the first layer
        self.layers[0].set_output(x, is_training)
        
        # set output for other layers
        for i in range(1, len(self.layers)):
            self.layers[i].set_output(self.layers[i-1].output, is_training)
              
        return self.layers[-1].output 
    
    
    
    def cost(self, logits, y):
        """
        computer total loss -- l2_loss + softmax cross entropy loss
        labels -- y
        logits -- uscaled (before apply softmax) output logits of the 
        
        return: the loss tensor
        """
        labels = tf.cast(y, tf.float32)
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        
        tf.add_to_collection('losses', cross_entropy_mean)
        
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
        #Add scalar summary for the loss
        loss_summary = tf.summary.scalar('loss_summary', loss)
                
        return loss, loss_summary
        
    
    def evaluation(self, logits, y):
        
        y = tf.argmax(y, axis=1)
        _, prediction = tf.nn.top_k(logits, k=1, name='prediction')
        
        correct = tf.nn.in_top_k(logits, y, k=1)
        
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
        
        accuracy_summary = tf.summary.scalar('accuracy_summary', accuracy)
        
        return prediction, accuracy, accuracy_summary
    
    
          
    def optimize(self, loss, learning_rate, global_step):
        
        #Create Adam optimizer        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer_op = optimizer.minimize(loss,global_step=global_step)
        
        #Track moving average of trained parameters
        var_averages = tf.train.ExponentialMovingAverage(
                0.999, global_step)
        
        var_averages_op = var_averages.apply(tf.trainable_variables())
        
        with tf.control_dependencies([optimizer_op, var_averages_op]):
            train_op = tf.no_op(name='train')
        
        return train_op
    
    
