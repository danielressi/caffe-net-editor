# caffe-net-editor
API to modify existing caffe nets for transfer learning

This work was created due to the lack of pycaffe documentation and missing contributions for automated editing of caffe model definitions in Python. You can use this API to modify an existing model definition (usually *-deploy.prototxt) into a model for transfer learning, ie. freeze layers, edit layers, remove and add new layers. In contrast to caffe.NetSpec() it modifies prototxt files directly. Consequently there is no need to tediously recreate existing architectures layer by layer in pycaffe. 

# Usage
Load existing caffe net (.prototxt) into the 
