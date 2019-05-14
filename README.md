# Prototypical Networks for Few-shot Learning
It is a re-implementation for paper "Prototypical Networks for Few-shot Learning" in PyTorch.

Experiment requirements:
* Python3
* torch 1.0
* torchvision 0.2.2


It mainly contains two parts:
* omniglotPy: train model and test in omniglot dataset
* miniImagePy:  train model and test in miniImagenet dataset
* myutils.py: is some tools that would be used



## omniglotPy
All categories of images are stored separately in each folder.

File structure:
* loadData.py: is used to load data in episode form;
* protoNet.py: is used to create a prototypical network model;
* train.py: is used to train the model;
* test.py: is used to evaluation;


Our results model realized:
* On omniglot dataset with 60-way 
  * 5-shot: 99.59% (99.7% in paper)
  
the results stored in 'nohup.out'
the best model was stored in .saved/min-loss.pth



## miniImagePy
All images are stored in a folder.

File structure:
* miniImageProcess.py: is used to preprocess dataset. converting the image data to pre-trained ndarray data and stored in two different ways.
* protoNet.py: is used to create a prototypical network model;
* loadData_sampler1.py:  sampler episode data like omniglot way when all categories of images are stored separately in each folder;
* train_mini_sampler1.py: train the model by using sampler1's data;
* train_mini_sampler1_.out:  30-way, 5-shot, training's outputs
* loadData_sampler2.py:  sampler episode data when all images are stored in a folder;
* train_mini_sampler2.py: train the model by using sampler2's data;
* train_mini_sampler2_.out:  30-way, 1-shot, training's outputs


> note that: actually sampler1 and sampler2 are equal. Just loading differently.


Our model realized:
* On miniImagenet dataset with 30-way 
  * 1-shot: 47.01% (49.42% in paper)  the best model was stored in .saved/min-loss.pth
  * 5-shot: 61.07% (68.20% in paper)  the best model was stored in .saved_train_mini/min-loss.pth
  
  


