# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:31:15 2017

The program creates a tensorflow convolutional neural net and train it with 
the MNIST data set using multiple GPUs.

cnn architecture -- modified LeNet-5 (with batch normalization)  

@author: leiming.wang
"""
import tensorflow as tf
import numpy as np
import tqdm
import os
from convolutional_nn import *
import mnist_input
import argparse
tf.flags._global_parser = argparse.ArgumentParser()


FLAGS = tf.app.flags.FLAGS

   
tf.app.flags.DEFINE_string('data_dir',
                           os.path.join(os.path.dirname(
                                   os.path.realpath(__file__)),'data'),
                           """ training data dir.""")

tf.app.flags.DEFINE_string('train_dir',
                           os.path.join(os.path.dirname(
                                   os.path.realpath(__file__)),
                           'train_multi_gpu'),
                           """ directory to put log and checkpoint files.""")


tf.app.flags.DEFINE_integer('batch_size', 110, """ mini batch size.""")

tf.app.flags.DEFINE_integer('max_steps', 12500, """ # of training steps.""")

tf.app.flags.DEFINE_float('learning_rate', 0.1, """ initial learning rate.""")

tf.app.flags.DEFINE_integer('num_gpus', 2, """numer of gpus to use.""")


#Global constants
NUM_CLASSES = 10 
NUM_SAMPLES = 55000 
NUM_EPOCHS_PER_DECAY = 1.0 
LEARNING_RATE_DECAY_RATE = 0.9
TOWER_NAME = 'hlus_hinton_gpu' 



def tower_loss(scope, model, images, labels, is_training):
    """Calculate the total loss on a single GPU
    Args:
        scope: unique prefix string identifying the running gpu, e.g. 'gpu_0'
    """
    logits = model.inference(images, is_training)
    _, __ = model.cost(logits, labels)
    
    #Assemble the losses for the current gpu only.
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
        
    
    loss_summary = tf.summary.scalar('loss_summary', total_loss)
    
    return total_loss, loss_summary
            

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all
    towers. This provides a synchronization point across all tower.
    Args:
        tower_grads: list of lists of (gradient, variable) tuples. The outer
        list is over individual gradients, and the inner list is over the
        gradient calculation for each tower.
    Return:
        List of (gradient, variable) where the gradient has been averaged 
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like:
        # ((grad0_gpu0,var0_gpu0), ..., (grad0_gpuN, vars_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add another dimension to the gradients to represent the tower
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
            
        # Average over the 'tower' dimension
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)
        
        # Variables are redundant since they are shared across towers.
        # so just return the first tower's pointer to the Variable
        v = grad_and_vars[0][1]
        
        ave_grad_and_var = (grad, v)
        
        average_grads.append(ave_grad_and_var)
    
    return average_grads


def mnist_train_on_multiple_gpus():
    
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
                
        is_training = tf.placeholder_with_default(True, [], name='is_training')
        #sub-graph control bool for the batchnorm layers and dropout layers  
        
        """ Create network structure to build computation graph""" 
        model = Network([ConvLayer('conv1',
                                 input_shape=[-1, 28, 28, 1],
                                 filter_shape=[5, 5, 1, 6],
                                 strides = [1, 1, 1, 1],
                                 padding='SAME',
                                 weight_decay=0.0),
                       BatchNormLayer('bn1', input_shape=[-1, 28, 28, 6]),
                       ReluLayer('relu1'),
                       MaxPoolLayer('pool1',
                                    ksize=[1,2,2,1],
                                    strides=[1,2,2,1],
                                    padding='SAME'),
                       ConvLayer('conv2',
                                 input_shape=[-1, 14, 14, 6],
                                 filter_shape=[5, 5, 6, 16],
                                 strides=[1,1,1,1],
                                 padding='VALID', 
                                 weight_decay=0.0),
                       BatchNormLayer('bn2', input_shape=[-1, 10, 10, 16]),
                       ReluLayer('relu2'),
                       MaxPoolLayer('pool2',
                                    ksize=[1,2,2,1],
                                    strides=[1,2,2,1],
                                    padding='SAME'),
                       ConvLayer('conv3',
                                 input_shape=[-1, 5, 5, 16],
                                 filter_shape=[5, 5, 16, 120],
                                 strides=[1, 1 ,1, 1],
                                 padding='VALID',
                                 weight_decay=0.0),
                       BatchNormLayer('bn3', input_shape=[-1, 1, 1, 120]),
                       ReluLayer('relu3'),
                       DropOutLayer('dropout3', keep_prob=0.7),
                       FullyConnectedLayer('full4',
                                           input_shape=[-1, 120],
                                           output_shape=[-1, 84],
                                           weight_decay=0.0),
                       BatchNormLayer('bn4', input_shape=[-1, 84]),
                       ReluLayer('relu4'),
                       DropOutLayer('dropout4', keep_prob=0.5),
                       FullyConnectedLayer('full5',
                                           input_shape=[-1, 84],
                                           output_shape=[-1,10],
                                           weight_decay=0.0)])
        
       
                
        # Create a exponential decay learning rate, and an optimizer
        num_batchs_per_epoch = NUM_SAMPLES / FLAGS.batch_size
        decay_steps = int(num_batchs_per_epoch * NUM_EPOCHS_PER_DECAY)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   LEARNING_RATE_DECAY_RATE,
                                                   staircase=True)
                 
        opt = tf.train.AdamOptimizer(learning_rate)
        
        # Get input
        images, labels = mnist_input.load_batch(FLAGS.data_dir, FLAGS.batch_size)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2*FLAGS.num_gpus)
        
        # Calculate the gradients for each tower
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' %i):
                    with tf.name_scope('%s_%d' %(TOWER_NAME, i)) as scope:                        
                         
                        image_batch, label_batch = batch_queue.dequeue()
                        # Calculate the loss for one tower of the model
                        # and retain the loss summary from the last tower
                        loss, loss_summary = tower_loss(scope, model,
                                                        image_batch,
                                                        label_batch,
                                                        is_training)

                        #Reuse variables for the next tower
                        tf.get_variable_scope().reuse_variables()
                                                
                        # Calculate the gradients for the batch data on this
                        # tower
                        grads = opt.compute_gradients(loss)
                        
                        # Keep track of the gradients across all towers
                        tower_grads.append(grads)
        
        
        # Calcuate the mean of tower gradients. This is the synchronization
        # point across all towers.
        grads = average_gradients(tower_grads)
        
        
        # Apply gradient to optimize the variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        
        # Track the moving average of trainable variables
        var_averages = tf.train.ExponentialMovingAverage(0.999, global_step)
        var_averages_op = var_averages.apply(tf.trainable_variables())
        
        # Group all updates into a single train_op
        train_op = tf.group(apply_gradient_op, var_averages_op)
        
        #define initializer and saver
        init = tf.global_variables_initializer()
            
        saver = tf.train.Saver()
        
        
        """ Create a session to run the training """
        # allow_soft_placement must be set to True as some of the ops do not
        # have GPU implementations
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True)) as sess:
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            
            sess.run(init)
            
            for step in tqdm.tqdm(range(FLAGS.max_steps)):
                
                sess.run(train_op)
                
                if step % 500 == 0: # write summary and print overview
                    
                    batch_loss, batch_loss_summ = \
                                        sess.run([loss, loss_summary])
                    
                    print('Step %d: batch_loss = %.3f' % (step, batch_loss))                                  
                    
                    # Write training summary
                    summary_writer.add_summary(batch_loss_summ,
                                               global_step=step)
                    
            
            # Stop the queueing threads
            coord.request_stop()
            # ... and we wait for them to do so before releasing the main thread
            coord.join(threads)
            #Flush the event file to disk and close file
            summary_writer.close()
            
            save_path = os.path.join(FLAGS.train_dir, 'mnist') 
            saver.save(sess, save_path)
            print('Model saved in file %s' % save_path) 
            



def main(argv=None): # pylint: disable=unused-argument

    
    mnist_train_on_multiple_gpus()



if __name__ == '__main__':
    
    tf.app.run()    