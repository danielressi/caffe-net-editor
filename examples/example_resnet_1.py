#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 00:56:47 2017

@author: danielressi
"""
from proto_editor import ProtoNetEditor
import caffe
from caffe import layers as L
from caffe import params as P


# Set paths for models and data sources
model_file_orig = './caffe-nets/ResNet-50-deploy.prototxt'
model_file_new = './caffe-nets/ResNet-50-train-val.prototxt'
lmdb_train = './path/to/train_lmdb'
lmdb_valid = './path/to/valid_lmdb'


mean_value = [150,150,150] # mean intensity values of training set (b,g,r)
# Create caffe Data Layers for training and validation
transform_param =  dict(crop_size = 224,mean_value = mean_value)      
data_train = L.Data(name='data1',top=['data','label'],batch_size = 32, 
                    backend = P.Data.LMDB,source = lmdb_train, 
                    transform_param = transform_param,
                    include = {'phase':caffe.TRAIN})

data_valid = L.Data(name='data2',top=['data','label'],batch_size = 32, 
                    backend = P.Data.LMDB,source = lmdb_valid, 
                    transform_param = transform_param,
                    include = {'phase':caffe.TEST,'stage':'test-on-test'})

data_train_test = L.Data(name='data3',top=['data','label'],batch_size = 32, 
                    backend = P.Data.LMDB,source = lmdb_train, 
                    transform_param = transform_param,
                    include = {'phase':caffe.TEST,'stage':'test-on-train'})

# To use model in python.
# memory_data = L.MemoryData(name = 'data',top='label',
#                                batch_size = 1, height = 224, width = 224,
#                                channels = 3)


net = ProtoNetEditor('myResNet')

net.putLayer(data_train)
net.putLayer(data_valid)
net.putLayer(data_train_test)

net.putModel(model_file_orig, auto_freeze = True)
net.popLayer()

# Change classifier
net.editLayer(name = 'fc1000',new_name = 'fc3',num_output = 3)

# Add loss and accuracy layer
loss = L.SoftmaxWithLoss(name = 'loss',bottom = ['fc3','label'])
net.putLayer(loss)

accuracy = L.Accuracy(name='accuracy',bottom = ['fc3','label'],include = {'phase':caffe.TEST})
net.putLayer(accuracy)

# Save new model    
net.save(model_file_new) 
