# caffe-net-editor
API to modify existing caffe nets for transfer learning

This work was created due to the lack of pycaffe documentation and missing contributions for automated editing of caffe model definitions in Python. You can use this API to transform an existing model definition (usually *-deploy.prototxt) into a model for transfer learning, ie. freeze layers, edit layers, remove and add new layers. In contrast to caffe.NetSpec() it modifies prototxt files directly. Consequently there is no need to tediously recreate existing architectures layer by layer in pycaffe. 

Initially this API was developed to make ResNet-50 model ready for transfer learning (there is little information on the web on how to do this successfully. This should help. The example.py uses the ResNet architecture.)

# Usage
* Initialize a new protoEditor, with a name for the new net
* Load existing caffe net definiton (.prototxt) with putModel()
* Add / modify Layers
* Save new net definition to file

# Note: 

* Layers can only be stacked (i.e. insertion at specific locations is not implemented yet)
* Not compatible for old caffe definitions
