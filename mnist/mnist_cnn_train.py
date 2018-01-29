# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:31:15 2017

The program creates a tensorflow convolutional neural net and train it with 
the MNIST data set.

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
                                   os.path.realpath(__file__)),'train'),
                           """ directory to put log and checkpoint files.""")


tf.app.flags.DEFINE_integer('batch_size', 110, """ mini batch size.""")

tf.app.flags.DEFINE_integer('max_steps', 25000, """ # of training steps.""")

tf.app.flags.DEFINE_float('learning_rate', 0.1, """ initial learning rate.""")

#Global constants
NUM_CLASSES = 10 
NUM_SAMPLES = 55000 
NUM_EPOCHS_PER_DECAY = 1.0 
LEARNING_RATE_DECAY_RATE = 0.9 



def mnist_train():
    
    with tf.Graph().as_default():
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
                
        """
        get training data from a running queue
        """
        images, labels = mnist_input.load_batch(FLAGS.data_dir, FLAGS.batch_size)
               
        
        is_training = tf.placeholder_with_default(True, [], name='is_training')
        #sub-graph control bool for the batchnorm layers and dropout layers 
        
        """ Create network structure and build computation graph""" 
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
        
        model.build_graph(images, labels, is_training)
        
        # Create a exponential decay learning rate, and build the trian_step
        num_batchs_per_epoch = NUM_SAMPLES / FLAGS.batch_size
        decay_steps = int(num_batchs_per_epoch * NUM_EPOCHS_PER_DECAY)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   LEARNING_RATE_DECAY_RATE,
                                                   staircase=True)
          
        train_step = model.optimize(model.loss, learning_rate, global_step)
        
        #define initializer and saver
        init = tf.global_variables_initializer()
            
        saver = tf.train.Saver()
        
        
        """ Create a session to run the training """
        with tf.Session() as sess:
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            
            sess.run(init)
            
            for step in tqdm.tqdm(range(FLAGS.max_steps)):
                
                sess.run(train_step)

                if step % 500 == 0: # write summary and print overview
                    
                    batch_loss, batch_loss_summ = \
                    sess.run([model.loss, model.loss_summary])

                    print('Step %d: batch_loss = %.4f' % (step, batch_loss))                                  
                    
                    # write the training summary to file
                    summary_writer.add_summary(batch_loss_summ,
                                               global_step=step)

            # Stop the queueing threads
            coord.request_stop()
            # ... and wait for them to do so before releasing the main thread
            coord.join(threads)
            #Flush the event file to disk and close file
            summary_writer.close()
            
            # Save the trained model
            save_path = os.path.join(FLAGS.train_dir, 'mnist') 
            saver.save(sess, save_path)
            print('Model saved in file %s' % save_path) 
            

def main(argv=None): # pylint: disable=unused-argument

    
    mnist_train()



if __name__ == '__main__':
    
    tf.app.run()    
