"""
An example of running convolutional neural network. 

Yifeng Li
CMMT, UBC, Vancouver
Sep. 23, 2014
Contact: yifeng.li.cn@gmail.com
"""
import os
import numpy

import convolutional_mlp
import classification as cl
from gc import collect as gc_collect

numpy.warnings.filterwarnings('ignore')

path="/home/yifeng/YifengLi/Research/deep/extended_deep/v1_0/"
os.chdir(path)

# load data
"""
A data set includes three files: 

[1]. A TAB seperated txt file, each row is a sample, each column is a feature. 
No row and columns allowd in the txt file.
If an original sample is a matrix (3-way array), a row of this file is actually a vectorized sample,
by concatnating the rows of the original sample.

[2]. A txt file including the class labels. 
Each row is a string (white space not allowed) as the class label of the corresponding row in [1].

[3]. A txt file including the name of features.
Each row is a string (white space not allowed) as the feature name of the corresponding column in [1].
"""

##################################
#load your data here ...
##################################

rng=numpy.random.RandomState(1000)    
numpy.warnings.filterwarnings('ignore')        
# train
classifier,training_time=convolutional_mlp.train_model( train_set_x_org=train_set_x_org, train_set_y_org=train_set_y_org,
                        valid_set_x_org=valid_set_x_org, valid_set_y_org=valid_set_y_org, 
                        n_row_each_sample=4,
                        learning_rate=0.1, alpha=0.01, n_epochs=1000, rng=rng, 
                        nkerns=[4,4,8],batch_size=500,
                        receptive_fields=((2,8),(2,8),(2,2)),poolsizes=((1,8),(1,8),(1,2)),full_hidden=8)

# test
test_set_y_pred=convolutional_mlp.test_model(classifier,test_set_x_org)

# evaluate classification performance
perf,conf_mat=cl.perform(test_set_y_org,test_set_y_pred,numpy.unique(train_set_y_org))
print perf
print conf_mat

# collect garbage
gc_collect()
