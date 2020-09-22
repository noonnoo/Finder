#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:18:13 2019

@author: onee
"""

from scipy import misc
import tensorflow as tf
import numpy as np
import os
import copy
import facenet
import align.detect_face

def make_img_emb(image):
    model = '../20180402-114759/'
    image_size = 160    
    margin = 44
    gpu_memory_fraction = 1.0
    
    image = [image] # ['/Users/onee/face_recognition/facenet_cw/test_img/Adrianne Palicki.jpg'] #나중에 변수로 받기(검색할 이미지
    image = load_and_align_data(image, image_size, margin, gpu_memory_fraction)
    
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            search_feed_dict = { images_placeholder: image, phase_train_placeholder:False }
            image = sess.run(embeddings, feed_dict=search_feed_dict)[-1]
    
    image = image.reshape((1,len(image)))
    
    return image


def make_folder_emb(image_path):
    model = '../20180402-114759/'
    image_size = 160    
    margin = 44
    gpu_memory_fraction = 1.0
        
    image_files = []
    person_name = []
    for i in os.listdir(image_path):
        if not i == '.DS_Store':
            for f in os.listdir(os.path.join(image_path, i)):
                ext = os.path.splitext(f)[1]
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                     image_files.append(os.path.join(image_path, i, f))
                     person_name.append(i)

    images = load_and_align_data(image_files, image_size, margin, gpu_memory_fraction)
#    images = load_and_align_data2(image_files, image_size, margin, gpu_memory_fraction)
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
                        
            
    return emb, person_name

def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

from keras.preprocessing.image import ImageDataGenerator

def load_and_align_data2(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        
        #data augmentation
        datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
#                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.1,
                horizontal_flip=False,
                fill_mode='nearest')
        datagen.fit(aligned)
        
        i = 0
        it = datagen.flow(aligned, batch_size=1)
        for i in range(5):
            batch = it.next()
            b = batch[0]
            print(b)
        for aug_img in datagen.flow(aligned):
            aug_img = aug_img.reshape(aug_img.shape[1],aug_img.shape[2],aug_img.shape[3])
            prewhitened = facenet.prewhiten(aug_img)
            img_list.append(prewhitened)
#            b.append(i)
            if i == 5:
                break
            i = i + 1
            
    images = np.stack(img_list)
    return images