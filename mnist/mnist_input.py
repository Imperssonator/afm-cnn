# -*- coding: utf-8 -*-
"""
Created on Tue Jan  18 09:27:57 2018
Create input pipeline for the mnist cnn model

@author: leiming.wang
"""

import tensorflow as tf

IMAGE_SIZE = 28*28
NUM_CLASSES = 10



def load_batch(data_dir, batch_size):
    
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    
    min_samples_after_dequeue = 22000
    
    q = tf.FIFOQueue(capacity=min_samples_after_dequeue,
                     dtypes=[tf.float32, tf.int32], shapes=[[IMAGE_SIZE],
                                                            [NUM_CLASSES]])
    
    enqueue_op = q.enqueue_many([mnist.train.images, mnist.train.labels])
    
    num_threads = 1
    
    qr = tf.train.QueueRunner(q, [enqueue_op]*num_threads)
    tf.train.add_queue_runner(qr)
    
    image, label = q.dequeue()
    
    batch_images, batch_labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_samples_after_dequeue + 3 * batch_size,
            min_after_dequeue=min_samples_after_dequeue)


    return batch_images, batch_labels


"""
with tf.Session() as sess:
    
    coord = tf.train.Coordinator()
    
    threads = tf.train.start_queue_runners(coord=coord)
    
    for i in range(5):
        print(sess.run([batch_images, batch_labels])) 
    # Request the child threads to stop ...
    coord.request_stop()
    # ... and wait for them to do so before releasing the main thread
    coord.join(threads)
"""


