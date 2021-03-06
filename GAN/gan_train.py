#!/usr/bin/env python
import os
from glob import glob
import random
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import click
import tensorflow as tf
import time


def get_image(image_path):
    """
    Input: image_path, string
    Return: img, (height x width x channels) np.uint8
    """
    
    img = io.imread(image_path)

    return img


def get_batch(image_files):
    """
    Input: image_files, list of strings
    Return: data_batch, (#images x height x width x channels) np.float32 (still 1-255 though)
    """
    data_batch = np.array(
        [get_image(sample_file) for sample_file in image_files]).astype(np.float32)

    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch


def get_batches(data_files, batch_size):
    """
    Input: batch_size, int
    Return: generator that yields image batches scaled to [-1,1]
    with shape (#images x height x width x channels) np.float32
    """
    IMAGE_MAX_VALUE = 255

    current_index = 0
    while current_index + batch_size <= len(data_files):
        data_batch = get_batch(
            data_files[current_index:current_index + batch_size])

        current_index += batch_size

        yield data_batch / IMAGE_MAX_VALUE * 2 - 0.5

        
def n_imshow(img_list):
    
    n = len(img_list)
    f = plt.figure(figsize=(10,3))
    
    for i in range(n):
        plt.subplot(1,n,i+1), plt.imshow(np.squeeze(img_list[i]),cmap='viridis')
        
    return f

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    """
    inputs_real = tf.placeholder(tf.float32, shape=(None, image_width, image_height, image_channels), name='input_real') 
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    return inputs_real, inputs_z, learning_rate


def discriminator(images, reuse=False, net={}):
    """
    Create the discriminator network
    """
    alpha = 0.2
    
    with tf.variable_scope('discriminator', reuse=reuse):
        # using 4 layer network as in DCGAN Paper
        
        # Conv 1
        net['conv1'] = tf.layers.conv2d(images, 32, 5, 2, 'SAME')
        net['lrelu1'] = tf.maximum(alpha * net['conv1'], net['conv1'])
        
        # Conv 2
        net['conv2'] = tf.layers.conv2d(net['lrelu1'], 64, 5, 2, 'SAME')
        net['batch_norm2'] = tf.layers.batch_normalization(net['conv2'], training=True)
        net['lrelu2'] = tf.maximum(alpha * net['batch_norm2'], net['batch_norm2'])
        
        # Conv 3
        net['conv3'] = tf.layers.conv2d(net['lrelu2'], 128, 5, 2, 'SAME')
        net['batch_norm3'] = tf.layers.batch_normalization(net['conv3'], training=True)
        net['lrelu3'] = tf.maximum(alpha * net['batch_norm3'], net['batch_norm3'])
        
        # Conv 4
        net['conv4'] = tf.layers.conv2d(net['lrelu3'], 256, 5, 2, 'SAME')
        net['batch_norm4'] = tf.layers.batch_normalization(net['conv4'], training=True)
        net['lrelu4'] = tf.maximum(alpha * net['batch_norm4'], net['batch_norm4'])
       
        # Flatten
        net['flat'] = tf.reshape(net['lrelu4'], (-1, 16*16*256))
        net['dropout'] = tf.layers.dropout(net['flat'], rate=0.4)
        
        # Logits
        net['logits'] = tf.layers.dense(net['dropout'], 1)
        
        # Output
        net['out'] = tf.sigmoid(net['logits'])
        
        return net['out'], net['logits'], net
    
    
def generator(z, out_channel_dim, is_train=True, net={}):
    """
    Create the generator network
    """
    alpha = 0.2
    
    with tf.variable_scope('generator', reuse=False if is_train==True else True):
        # First fully connected layer
        net['x_1'] = tf.layers.dense(z, 16*16*256)
        
        # Reshape it to start the convolutional stack
        net['deconv_2'] = tf.reshape(net['x_1'], (-1, 16, 16, 256))
        net['batch_norm2'] = tf.layers.batch_normalization(net['deconv_2'], training=is_train)
        net['lrelu2'] = tf.maximum(alpha * net['batch_norm2'], net['batch_norm2'])
        
        # Deconv 1
        net['deconv3'] = tf.layers.conv2d_transpose(net['lrelu2'], 256, 5, 2, padding='SAME')
        net['batch_norm3'] = tf.layers.batch_normalization(net['deconv3'], training=is_train)
        net['lrelu3'] = tf.maximum(alpha * net['batch_norm3'], net['batch_norm3'])
        
        # Deconv 2
        net['deconv4'] = tf.layers.conv2d_transpose(net['lrelu3'], 128, 5, 2, padding='SAME')
        net['batch_norm4'] = tf.layers.batch_normalization(net['deconv4'], training=is_train)
        net['lrelu4'] = tf.maximum(alpha * net['batch_norm4'], net['batch_norm4'])
        
        # Deconv 3
        net['deconv5'] = tf.layers.conv2d_transpose(net['lrelu4'], 64, 5, 2, padding='SAME')
        net['batch_norm5'] = tf.layers.batch_normalization(net['deconv5'], training=is_train)
        net['lrelu5'] = tf.maximum(alpha * net['batch_norm5'], net['batch_norm5'])
        
        # Output layer (Deconv 3)
        net['logits'] = tf.layers.conv2d_transpose(net['lrelu5'], out_channel_dim, 5, 2, padding='SAME')
        
        net['out'] = tf.tanh(net['logits'])
        
        return net['out'], net
    
    
def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    """
    
    label_smoothing = 0.9
    
    g_model, _ = generator(input_z, out_channel_dim)
    d_model_real, d_logits_real, _ = discriminator(input_real)
    d_model_fake, d_logits_fake, _ = discriminator(g_model, reuse=True)
    
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_model_real) * label_smoothing))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.zeros_like(d_model_fake)))
    
    d_loss = d_loss_real + d_loss_fake
                                                  
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.ones_like(d_model_fake) * label_smoothing))
    
    
    return d_loss, g_loss


def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    """
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


def show_generator_output(sess, n_images, input_z, out_channel_dim):
    """
    Show example output for the generator
    """
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples, _ = sess.run(
        generator(input_z, out_channel_dim, is_train=False),
        feed_dict={input_z: example_z})
    
    num_to_show = max(n_images,6)
    
    f = n_imshow(samples[:num_to_show])
    return f


def example_generator_output(sess, n_images, input_z, out_channel_dim):
    """
    Show example output for the generator
    """
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples, _ = sess.run(
        generator(input_z, out_channel_dim, is_train=False),
        feed_dict={input_z: example_z})
    
    return samples

    
@click.command()
@click.argument('datadir', type=click.Path())
@click.option('--epoch-count', '-e', type=int, default=2)
@click.option('--batch-size', '-s', type=int, default=10)
@click.option('--z-dim', '-z', type=int, default=100)
@click.option('--learning-rate', '-r', type=float, default=0.0002)
@click.option('--beta1', '-b', type=float, default=0.5)
def train(datadir, epoch_count=2, batch_size=10, z_dim=100, learning_rate=0.0002, beta1=0.5):
    """
    Train the GAN
    """
    
    # Make directory to save figures
    fig_save_dir = os.path.join(os.path.split(datadir)[0],'figures')
    try:
        os.makedirs(fig_save_dir)
    except FileExistsError:
        pass
    
    
    # Build data file list and determine shape
    data_files = glob(os.path.join(datadir, '*.png'))
    random.shuffle(data_files)  # This way the batches won't bias the training steps toward any user
    test_img = get_image(data_files[0])
    IMAGE_WIDTH = test_img.shape[0]
    IMAGE_HEIGHT = test_img.shape[1]
    if len(test_img.shape) < 3:
        IMAGE_CHANS = 1
    else:
        IMAGE_CHANS = test_img.shape[2]
        
    data_shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANS
    
    
    # Initialize input tensors and model graph
    input_real, input_z, _ = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3])
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
    
    with tf.Session() as sess:
        
        # Initialize all global variables
        sess.run(tf.global_variables_initializer())
        
        for epoch_i in range(epoch_count):
            
            steps = 0
            steps_per_epoch = int(data_shape[0]/batch_size)
            
            for batch_images in get_batches(data_files,batch_size):
                
                start = time.time()
                
                print('step {} of {}'.format(steps,steps_per_epoch))
                
                # values range from -0.5 to 0.5, therefore scale to range -1, 1
                steps += 1
            
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                
                _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(g_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                
                if steps in [int(steps_per_epoch/2), steps_per_epoch]:
                    # At halfway and final step in epoch, print loss and save generator images
                    train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
                    train_loss_g = g_loss.eval({input_z: batch_z})

                    print("Epoch {}/{}...".format(epoch_i+1, epoch_count),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    
                    # And save example generator images
                    num_gen_examples = 8
                    samples = example_generator_output(sess, num_gen_examples, input_z, data_shape[3])
                    print(samples.shape)
                    for i in range(num_gen_examples):
                        io.imsave(os.path.join(fig_save_dir,
                                               'gen_out_e{}_s{}_{}.png'.format(epoch_i,steps,i)),
                                  samples[i,:,:,0])
            
                end = time.time()
                print('step time was {}'.format(end-start))
                    

                                
if __name__ == '__main__':
    tf.reset_default_graph()
    with tf.Graph().as_default():
        train()
