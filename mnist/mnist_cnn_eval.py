# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:31:15 2017
The program re-build the tf graph same as the training graph,
and restore its variables from an saved tf model, 
for evalution or re-training on new data.   

@author: leiming.wang
"""
import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
from convolutional_nn import *
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

IMAGE_SIZE = 28*28
NUM_CLASSES = 10

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


my_imgs = np.zeros((10,784)).astype(np.float32)
my_labels = np.zeros((10,10)).astype(np.float32)
for i in range(10):
    img = cv2.imread("my_mouse_writing/"+str(i)+".jpg", 0)
    img = img.flatten()
    my_imgs[i,:] = img/255
    my_labels[i,i] = 1


def mnist_eval():
    
    with tf.Graph().as_default():
       
        """Re-build the graph"""
        
        # Use placeholder to feed in evaluation data
        eval_images = tf.placeholder(tf.float32, shape=[None,IMAGE_SIZE])
                    
        eval_labels = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
        
        is_training = tf.placeholder_with_default(False, [], name='is_training') 
            
        """ Re-create network structure and build computation graph""" 
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
    
        model.build_graph(eval_images, eval_labels, is_training)
        
        """ Create a session to import saved model and restore the variabel """
        with tf.Session() as sess:
            
            # Restore the moving average version of the learned variables for eval
            variable_averages = tf.train.ExponentialMovingAverage(0.999)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            save_path = os.path.join(FLAGS.train_dir, 'mnist') 
            saver.restore(sess, save_path)
                        
            # prepare feed_dict            
            train_fd = {eval_images: mnist.train.images,
                        eval_labels: mnist.train.labels}
            
            validation_fd = {eval_images: mnist.validation.images,
                             eval_labels: mnist.validation.labels}
            
            test_fd = {eval_images: mnist.test.images,
                       eval_labels: mnist.test.labels}
            
            
            # Evaluate against the train data set
            train_accuracy = sess.run(model.accuracy,feed_dict = train_fd)                           
            print('train_accuracy = %.4f' % train_accuracy)
                    
            # Evaluate against the validation data set
            validation_accuracy = sess.run(model.accuracy,feed_dict = validation_fd)                           
            print('validation_accuracy = %.4f' % validation_accuracy)       
    
            # Evaluate against the test data set
            test_accuracy = sess.run(model.accuracy, feed_dict = test_fd)
            print('test_accuracy = %.4f' % test_accuracy)
            
            # Evaluate my mouse-written digits
            for my_img, my_label in zip(my_imgs, my_labels):
                predicted = print(sess.run(model.prediction,
                                           feed_dict = {eval_images:my_img.reshape(1,-1),
                                                        eval_labels: my_label.reshape(1,-1),
                                                        is_training: False}))


def main(argv=None): # pylint: disable=unused-argument

    
    mnist_eval()



if __name__ == '__main__':
    
    tf.app.run()  