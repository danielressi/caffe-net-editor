# caffe-net-editor
Pyhon API to modify existing caffe nets for transfer-learning / fine-tuning

This work was created due to the lack of pycaffe documentation and missing contributions for automated editing of caffe model definitions in Python. You can use this API to transform an existing model definition (for example a *-deploy.prototxt file) into a model for transfer learning, ie. freeze layers, edit layers, remove and add new layers. In contrast to caffe.NetSpec() it modifies prototxt files directly. Consequently there is no need to tediously recreate existing architectures layer by layer in pycaffe. 

Initially this API was developed to make the ResNet-50 model ready for transfer learning (there is little information on the web on how to do this successfully. Refer to ~/examples/example_resnet_1.py for a tutorial using the ResNet-50 architecture.)
The original ResNet model can be found and downloaded here:
https://github.com/KaimingHe/deep-residual-networks

# Install
In command line: change to directory ~/caffe-net-editor and execute "pip install setup.py"


# Usage
* from proto_editor import ProtoNetEditor
* Initialize a ProtoNetEditor with a name for the new net
* Load existing caffe net definiton (.prototxt) with putModel()
* Add / modify layers
* Save new net definition to file
* ...brew in caffe/pycaffe

# Note

* Layers can only be stacked (i.e. insertion at specific locations is not implemented yet)
* Not compatible for old caffe definitions

# Dependencies

* caffe (with pycaffe): http://caffe.berkeleyvision.org/installation.html
* Google protobuf
