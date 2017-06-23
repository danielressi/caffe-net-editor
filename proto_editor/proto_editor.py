#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 19:03:08 2017

@author: danielressi
"""

import caffe
import google.protobuf.text_format as txtf
from caffe import layers as L


def openCaffeSpec(proto_file):
    net_spec = caffe.proto.caffe_pb2.NetParameter()

    with open(proto_file) as f:
        s = f.read()
        txtf.Merge(s, net_spec)
        
    return net_spec
    
class ProtoNetEditor(object):
    """ API to modify existing caffe model (NetParameter).
    
        Use to modify a caffe "deploy.prototxt" file for transfer learning
        or to deploy a modified model.
    
        Steps (deploy.prototxt to train_val.prototxt):
            + initilaize with a name for the net
            + add a caffe data layer (see caffe.NetSpec())
            + add the layers of an existing prototxt model with putModel().
            + edit paramters (freeze layers, add new classifier)
            + add SoftmaxWithLoss layer
            + save new model to file (prototxt)
        
        Steps (train_val.prototxt ->   deploy.prototxt):
            + initilaize with a name for the net
            + add the layers of an existing prototxt model with putModel().
            + use deploy() to convert model.
            + save new model to file (prototxt)
    
    """    
    def __init__(self, name):
        """ Initialize editor.
        Creates an empty caffe NetParameter.
       
        Parameters
        ----------
        name: str
            Name of the new model. 
        """
        
        self.net_spec = caffe.NetSpec().to_proto()
        self.net_spec.name = name
        self.layer_types = None
        self.layer_names = None

    def _editParams(self, layer_ind, lr_mult_list=None, decay_mult_list=None):
        """ Edit learnable parameters of a layer. """
              
        num_params = len(self.net_spec.layer[layer_ind].param)          
        num_updates = len(lr_mult_list) \
                         if lr_mult_list is not None else len(decay_mult_list)
        
        
        if num_params == num_updates:
            for i in range(num_params):
                param = self.net_spec.layer[layer_ind].param[i]
                
                if lr_mult_list is not None: 
                    param.lr_mult = lr_mult_list[i]
                                
                if decay_mult_list is not None: 
                    param.decay_mult = decay_mult_list[i] 
        
        else:
            for i in range(num_params):
                self.net_spec.layer[layer_ind].param.pop()
    
            for lr_mult,decay_mult in zip(lr_mult_list,decay_mult_list):
                self.net_spec.layer[layer_ind].param.add(lr_mult=lr_mult,
                                                   decay_mult=decay_mult)
                
    def _updateLayers(self):
        """ Updates layer names and layer types in the editor. """
        
        self.layer_types = [l.type for l in self.net_spec.layer]
        self.layer_names = [l.name for l in self.net_spec.layer] 
        
    def _setDeployInput(self, input_dim):
        """ Set input attributes for deployment. """
        
        bottom_name = self.net_spec.layer[0].bottom[0]
        self.net_spec.input.append(bottom_name)
        
        if type(input_dim) is tuple:
            for dim in input_dim:
                self.net_spec.input_dim.append(dim)
        else:
            raise ValueError("input_dim needs to be a tuple indicating input shape!")
            
    def freezeAll(self):
        """ Freezes all learnable layers.
        
        Sets the learning rate multiplier and weight decay multiplier to 0.
        Ensures that these layers are held fixed during finetuning.
        """
        
        if self.layer_types is None:
            raise Exception("Error, no layers loaded yet!")
        

        for layer_ind, layer_type in enumerate(self.layer_types):
            if layer_type == 'Convolution' or layer_type == 'Scale':
                if len(self.net_spec.layer[layer_ind].param) == 0:              
                    if self.net_spec.layer[layer_ind].convolution_param.bias_term == True:
                        self.net_spec.layer[layer_ind].param.add(lr_mult= 0,decay_mult= 0)
                        self.net_spec.layer[layer_ind].param.add(lr_mult= 0,decay_mult= 0)
                    else:
                        self.net_spec.layer[layer_ind].param.add(lr_mult= 0,decay_mult= 0)                   

            if layer_type == 'BatchNorm':
                [self.net_spec.layer[layer_ind].param.add(lr_mult= 0) for x in xrange(3)]   
  
    
    def show(self, layer=None):
        ''' Prints model or single layer if specified.
        
            Parameters
            ----------
            layer: str
                Name of a 
        '''  
        
        if layer is None:
            return self.net_spec
        else:
            ind = self.layer_names.index(layer)
            return self.net_spec.layer[ind]
    def putLayer(self, Layer):
        ''' Put a new caffe layer at the top of the new net.
        
            Parameters
            ----------
            Layer: Caffe Layers object (caffe.net_spec.Layers)
                Must be a valid caffe.net_spec net specification.
                Further information use "help(caffe.net_spec)", refer to
                example.py, or refer to official caffe website:
                    http://caffe.berkeleyvision.org/tutorial/layers.html
        '''
        
        new_layer = self.net_spec.layer.add()
        new_layer.CopyFrom(Layer.to_proto().layer[0]) #,
        
        if len(new_layer.top) > 1:                           
            new_layer.top[0] = 'delete'
            new_layer.top.remove('delete')
            
        self._updateLayers()
        
    def putModel(self, filename_in, auto_freeze=True):
        ''' Add all layers of an existing net to the new net.
        
            Parameters
            ----------
            filename_in : str
                Path to the original model. 
            auto_freeze : Boolean
                If true all learnable will be are frozen.
        '''
        
        net_orig = openCaffeSpec(filename_in)
        if len(net_orig.layer) > 0:
            self.net_spec.layer.extend(net_orig.layer)
        elif len(net_orig.layers) > 0:
            raise Exception('Old proto format! Use "caffe/tools/\
            upgrade_net_proto_text.cpp" to upgrade')
        
        self._updateLayers()
        
        if auto_freeze == True:
            self.freezeAll()
            
    def popLayer(self, until=None):
        """ Remove Layer(s) from the top of the net.       
    
            Parameters
            ----------
            unitl : str
                Layer at which the continuous removal of layers is stopped.   
        """
        
        if until is None:
            self.net_spec.layer.pop()
           
        else:
            for layer_name in reversed(self.layer_names):
                if layer_name == until:
                    break
                else:
                    self.net_spec.layer.pop()
                   
        self._updateLayers() 
        
    def editLayer(self, name, new_name=None, num_output=None, lr_mult=None,
                          decay_mult=None, use_global_stats=None):
        """ Edit layer        
            
            Change name to initialize layer with random weights or edit
            the multiplier variables to freeze / unfreeze a layer.
            
            Parameters
            ----------
            name : str
                Name of the layer in the net. 
            new_name : str
                Name of the new layer.
            num_output: int
                Number of outputs.
            lr_mult: list
                Values of the learning rate multipliers (weight,bias) packed in
                a list. First value refers to weight, second to bias.
            decay_mult: list
                Values of the weight_decay multipliers (weight,bias) packed in
                a list. First value refers to weight, second to bias.
            use_global_stats: Boolean
                Only applicable for BatchNorm layers. By convention, use global
                stats is 
                
        """
        
        index = self.layer_names.index(name)
        layer = self.net_spec.layer[index]

        if new_name is not None:
            layer.name = new_name
            layer.top[0] = new_name
        if num_output is not None:
            layer.inner_product_param.num_output = num_output

        if lr_mult is not None or decay_mult is not None:
            self._editParams(index,lr_mult,decay_mult)
            
        if use_global_stats is not None and self.layer_types[index] == 'BatchNorm':
            layer.batch_norm_param.use_global_stats = use_global_stats
        self._updateLayers() 

    def deploy(self, input_dim=(1,3,224,224)):
        """ Turn training model into test model.
        
            Removes layers of type "Data","SoftmaxWithLoss" and "Accuracy".
            Removes parameters used for learning.
            Adds a Softmax layer to the model to compute class probabilities.
            
            Parameters
            ----------
            input_dim : tuple
                Values specifying input dimensions
                -> (Batch size,Number of Channels,Height,Width)    
        """
        
        trash_layers = ['Data','SoftmaxWithLoss','Accuracy']
        
        trash_bin = []
        for layer in self.net_spec.layer:
            
            if layer.type in trash_layers:
                trash_bin.append(layer)
                continue
            if len(layer.include) > 0:
                [layer.include.pop() for x in xrange(len(layer.include))]
            if len(layer.param) > 0:
                [layer.param.pop() for x in xrange(len(layer.param))]  
  
        for layer in trash_bin:
            self.net_spec.layer.remove(layer)
            
        clf = self.net_spec.layer[-1]
        assert clf.type == "InnerProduct"
        prob =  L.Softmax(name = 'prob',bottom = [clf.name])
        
        self.putLayer(prob)
        self._updateLayers()
        self._setDeployInput(input_dim)
        
    def save(self, filename_out):
        """ Save net to file.       
    
            Parameters
            ----------
            filename_out : str
                Path to write the new model to file (including .prototxt).   
        """   
        
        with open(filename_out, 'w') as f:
            f.write(str(self.net_spec))    
