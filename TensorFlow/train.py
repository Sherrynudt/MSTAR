#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:21:09 2017

@author: didizhang
"""

#%%

import os
import numpy as np
import tensorflow as tf
import input_data
import model

#%%

N_CLASSES = 3
IMG_W = 128  
IMG_H = 128
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001


#%%
def run_training():
    
    train_dir = '/Users/didizhang/Desktop/MSTAR/data/train/3_17_DEG/'
    logs_train_dir = '/Users/didizhang/Desktop/MSTAR/logs/train/'
    
    val_dir = '/Users/didizhang/Desktop/MSTAR/data/test/3_15_DEG/'
    logs_val_dir = '/Users/didizhang/Desktop/MSTAR/logs/val/'
    
    train, train_label = input_data.get_files(train_dir)
    val, val_label = input_data.get_files(val_dir)
    
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE, 
                                                          CAPACITY)    
    
    val_batch, val_label_batch = input_data.get_batch(val,
                                                      val_label,
                                                      IMG_W,
                                                      IMG_H,
                                                      BATCH_SIZE, 
                                                      CAPACITY)  
    
    logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    loss = model.losses(logits, train_label_batch)        
    train_op = model.trainning(loss, learning_rate)
    acc = model.evaluation(logits, train_label_batch)

    x = tf.placeholder(tf.float32,shape=[BATCH_SIZE, 96, 96, 1])
    y_ = tf.placeholder(tf.int16,shape=[BATCH_SIZE])
   
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        
        summary_op = tf.summary.merge_all()
        
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
   
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
            
            
                tra_images, tra_labels = sess.run([train_batch,train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                             feed_dict={x:tra_images,y_:tra_labels})
                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)  
                  
            
                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch,val_label_batch])
                    val_loss, val_acc = sess.run([loss,acc],
                                                 feed_dict={x:val_images,y_:val_labels})
                    print('** Step %d, val loss = %.2f, val accuracy = %.2f%% **' %(step, val_loss, val_acc*100.0))
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, step)               
                
            
                if step % 500 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        
    coord.join(threads)
    sess.close()
    

#%% Evaluate one image
# when training, comment the following codes.

"""
from PIL import Image
import matplotlib.pyplot as plt

def evaluate_one_image():
    #Test one image against the saved models and parameters

    train_dir = '/Users/didizhang/Desktop/MSTAR/data/train/3_17_DEG/'
    train, train_label = input_data.get_files(train_dir)
    
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = Image.open(img_dir) 
    image = image.resize([128, 128])
    #image = tf.random_crop(image, [96, 96, 1])# randomly crop the image size to 96 x 96
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image,lower=0.2,upper=1.8)
    
    plt.imshow(image)
   
    image = np.array(image)
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 3
        
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 96, 96, 1])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[96, 96, 1])

        logs_train_dir = '/Users/didizhang/Desktop/MSTAR/logs/train/' 
                      
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is BMP2 with possibility %.6f' %prediction[:, 0])
            if max_index==1:
                print('This is BTR70 with possibility %.6f' %prediction[:, 1])
            if max_index==2:
                print('This is T72 with possibility %.6f' %prediction[:, 2])
                """
#%%