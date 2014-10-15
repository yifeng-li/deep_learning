"""
A module for generic classification purpose.

Funtionality include:

normalize_l2norm: Normalize each row has unit l_2 norm.

normalize_mean0std1: Normalize each feature to have mean 0 and std 1.

balance_sample_size: Balance sample size of a data set among classes.  

change_class_labels: Change class labels to {0,1,2,3,...,C-1}.

change_class_labels_to_given: Change original class labels to a given labels.

merge_class_labels: Merge class labels into several super groups/classes.

take_some_classes: Only take sevaral classes, and remove the rest.

partition_train_valid_test: Partition the whole data into training, validation, and test sets.

reduce_sample_size: Reduce sample by to 1/times.

perform: Compute the classification performance.

write_feature_weight: Write the input layer weights to a file. 
                      Only applicable to deep feature selection.

write_feature_weight2: Write the input layer weights and other information to a file.
                       Only applicable to deep feature selection.

Yifeng Li
CMMT, UBC, Vancouver
Sep 23, 2014
Contact: yifeng.li.cn@gmail.com
"""
from __future__ import division
import numpy as np
from sklearn import cross_validation
import math

def normalize_l2norm(data):
    """
    Normalize each row has unit l_2 norm. 
    
    INPUTS:
    data: numpy 2d array or matrix, each row should be a sample.
    
    OUTPUTS:
    data: numpy 2d array or matrix, normalized data.
    
    Example:
    data=[[3,5,7,9],[3.0,2,1.1,8.4],[5.9,9,8,10]]
    data=np.array(data)
    data_normalized=normalize_l2norm(data)
    print data_normalized
    """
    data_sqrt=np.sqrt(np.square(data).sum(axis=1))
    data_sqrt.shape=(data_sqrt.shape[0],1)
    tol=2**-30
    data=(data+tol)/(data_sqrt+tol)
    return data
    
def normalize_mean0std1(data,data_mean=None,data_std=None):
    """
    Normalize each feature to have mean 0 and std 1.
    
    INPUTS:
    data: numpy 2d array or matrix, each row should be a sample.
    
    data_mean: numpy 1d array or vector, the given means of samples, useful for normalize test data.
    
    data_std: numpy 1d array or vector, the given standard deviation of samples, useful for normalize test data.
    
    OUTPUTS:
    data: numpy 2d array or matrix, normalized data.
    
    data_mean: numpy 1d array or vector, the given means of samples, useful for normalize test data.
    
    data_std: numpy 1d array or vector, the given standard deviation of samples, useful for normalize test data.
    """
    if data_mean is None:
        data_mean=np.mean(data,axis=0)
    data_mean.reshape((1,data_mean.shape[0]))
    if data_std is None:
        data_std=np.std(data,axis=0)
    data_std.reshape((1,data_std.shape[0]))
    tol=1e-16
    return (data-data_mean)/(data_std+tol),data_mean,data_std
    
def balance_sample_size(data,classes,others=None,min_size_given=None,rng=np.random.RandomState(100)):
    """
    Balance sample size of a data set among classes.
    
    INPUTS:
    data: numpy 2d array or matrix, each row should be a sample.
    
    classes: numpy 1d array or vector, class labels.
    
    others: numpy 2d array or matrix, extra information of samples if available,
    each row should associated to a row of data.
    
    min_size_given: int, the size of each class wanted.
    
    rng: numpy random state.
    
    OUTPUTS:
    data: numpy 2d array or matrix, each row should be a sample, balanced data.
    
    classes: numpy 1d array or vector, balanced class labels.
    
    others: numpy 2d array or matrix, balanced other information.
    
    Example:
    data=[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7]]
    data=np.array(data)
    classes=np.array(['zz','xx','xx','yy','zz','yy','xx'])
    balance_sample_size(data,classes)
    """    
    u, indices = np.unique(classes,return_inverse=True)
    indices=np.asarray(indices)
    num_u=len(u)
    sample_sizes=[]
    
    # get sample size of each class
    for i in xrange(num_u):
        sample_size_this=np.sum(indices==i)
        sample_sizes.append(sample_size_this)     
        
    size_min=np.amin(sample_sizes) # smallest sample size
    
    if min_size_given and size_min>min_size_given:
        size_min=min_size_given   
        
    indices_all=np.array([],dtype=indices.dtype)
    indices_range=np.array(range(len(indices)))
    
    for i in xrange(num_u):
        ind_this_num=indices_range[indices==i]
        ind_this_reduced=ind_this_num[rng.choice(len(ind_this_num),size=size_min,replace=False)]
        indices_all=np.append(indices_all,ind_this_reduced)
    
    # reduce the data    
    data=data[indices_all]
    classes=classes[indices_all]
    if others:
        others=others[indices_all]
    return data,classes,others
    
def change_class_labels(classes):
    """
    Change class labels to {0,1,2,3,...,C-1}.
    
    INPUTS:
    classes: numpy 1d array or vector, the original class labels.
    
    OUTPUTS:
    u: numpy 1d array or vector, the unique class labels of the original class labels.
    
    indices: numpy 1d array or vector, the new class labels from {0,1,2,3,...,C-1}.
    
    Example:
    classes=['c2','c3','c2','c1','c2','c1','c3','c2']
    change_class_labels(classes)
    Yifeng Li, in UBC
    Aug 22, 2014.
    """
    u,indices=np.unique(classes,return_inverse=True)
    return u,indices

def change_class_labels_to_given(classes,given):
    """
    Change original class labels to a given labels.
    
    INPUTS:
    classes: numpy 1 d array or vector, the original class labels.
    
    given: dic, pairs of old and new labels.
    
    OUTPUTS:
    classes_new: numpy 1 d array or vector, changed class labels.
    
    Example:
    classes=[1,2,0,0,2,1,1,2]
    given={1:"class1", 2:"class2", 0:"class0"}
    change_class_labels_to_given(classes,given)
    """
    classes=np.asarray(classes)
    classes_new=np.zeros(classes.shape,dtype=object)
    for i in given:
        classes_new[classes==i]=given[i]
    return classes_new
    
    

def merge_class_labels(classes,group):
    """
    Merge class labels into several super groups/classes.
    
    INPUTS:
    classes: numpy 1 d array or vector, the original class labels.
    
    group: tuple of tuples or lists, 
    group[i] indicates which original classes to be merged to the i-th super class.
    
    OUTPUTS:
    classes_merged: numpy 1 d array or vector, the merged class labels.
    If original labels are strings, they are concatenated by "+".
    If original lables are numbers, they are renumbered starting from 0.
    
    Example
    classes=[0,3,4,2,1,3,3,2,4,1,1,0,0,1,2,3,4,1]
    group=([0],[1,2],[3,4])
    merge_class_labels(classes,group)
    classes=['c2','c1','c0','c0','c1','c2','c1']
    group=(['c0'],['c1','c2'])
    merge_class_labels(classes,group)
    """
    classes=np.asarray(classes)
    if (classes.dtype != int) and (classes.dtype != 'int64') and (classes.dtype != 'int32'):
        classes_merged=np.zeros(classes.shape,dtype=object)
        for subgroup in group:
            subgroup_label='+'.join(subgroup)
            for member in subgroup:
                classes_merged[classes==member]=subgroup_label
    else: # int class labels
        classes_merged=np.zeros(classes.shape,dtype=int)
        for i in range(len(group)):
            subgroup=group[i]
            for member in subgroup:
                classes_merged[classes==member]=i
    
    return classes_merged
    
def take_some_classes(data,classes,given):
    """
    Only take sevaral classes, and remove the rest.
    
    INPUTS:
    data: numpy 2d array or matrix, each row is a sample, the original data.
    
    classes: numpy 1d array or vector, class labels, the original labels.
    
    given: numpy 1d array or vector, indicates which classes to be taken.
    
    OUTPUTS:
    data: numpy 2d array or matrix, each row is a sample, the taken data.
    
    classes: numpy 1d array or vector, class labels, the taken labels.
    """
    classes=np.asarray(classes)
    log_ind=np.zeros(classes.shape,dtype=bool)
    for i in range(len(given)):
        log_ind[classes==given[i]]=True
    classes=classes[log_ind]
    data=data[log_ind]
    return data,classes

def partition_train_valid_test(data,classes,ratio=(1,1,1)):
    """
    Partition the whole data into training, validation, and test sets.
    
    INPUTS:
    data: numpy 2d array or matrix, each row is a sample, the original data.
    
    classes: numpy 1d array or vector, class labels, the original labels.
    
    ratio, int tuple or list of length 3, (ratio_of_train_set,ratio_of_valid_set,ratio_test_set).
    
    OUTPUTS:
    train_set_x: data of training set.
    
    train_set_y: class labels of training set.
    
    valid_set_x: data of validation set.
    
    valid_set_y: class labels of validation set.
    
    test_set_x: data of test set.
    
    test_set_y: class labels of test set.
    
    Example:
    data=np.random.random((20,3))
    classes=np.array([0,2,2,2,0,0,1,1,0,0,0,2,2,2,0,0,1,1,0,0],dtype=int)
    train_set_x,train_set_y,valid_set_x,valid_set_y,test_set_x,test_set_y \
    =partition_train_valid_test(data,classes,ratio=(2,1,1))
    Yifeng Li, in UBC.
    August 22, 2014.    
    """
    k=sum(ratio) # ratio must be a vector of integers
    skf = cross_validation.StratifiedKFold(classes, n_folds=k)
    train_ind=np.array([],dtype=int)
    valid_ind=np.array([],dtype=int)
    test_ind=np.array([],dtype=int)
    count=0    
    for (tr,te) in skf:
        if count<ratio[0]:
            train_ind=np.append(train_ind,te)
            count=count+1
            continue
        if count>=ratio[0] and count <ratio[0]+ratio[1]:
            valid_ind=np.append(valid_ind,[te])
            count=count+1
            continue
        if count>=ratio[0]+ratio[1]:
            test_ind=np.append(test_ind,[te])
            count=count+1
            continue
    train_set_x=data[train_ind]
    train_set_y=classes[train_ind]
    valid_set_x=data[valid_ind]
    valid_set_y=classes[valid_ind]
    test_set_x=data[test_ind]
    test_set_y=classes[test_ind]
    return train_set_x,train_set_y,valid_set_x,valid_set_y,test_set_x,test_set_y

def perform(y,y_predicted,unique_classes):
    """
    Compute the classification performance.
    
    INPUTS: 
    y: numpy 1d array or vector, the actual class labels.
    
    y_predicted: numpy 1d array or vector, the predicted class labels.
    
    unique_classes: numpy 1d array or vector of length C (# classes), all unique actual class labels.
    
    OUTPUTS:
    perf: numpy 1d array or vector of length C+2, 
    [acc_0, acc_1, acc_{C-1}, accuracy, balanced accuracy].
    
    confusion_matrix: numpy 2d array of size C X C, confusion matrix.
    
    Example:
    y=[0,0,1,1,1,2,2,2,2]
    y_predicted=[0,1,1,1,2,2,2,0,1]
    perform(y,y_predicted,[0,1,2])
    Yifeng Li, in UBC.
    August 23, 2014.
    """
    y=np.asarray(y,dtype=int)
    y_predicted=np.asarray(y_predicted,dtype=int)
    
    numcl=len(unique_classes)
    confusion_matrix=np.zeros((numcl,numcl),dtype=float)
    for i in xrange(len(y)):
        confusion_matrix[y[i],y_predicted[i]]=confusion_matrix[y[i],y_predicted[i]]+1
    perf=np.zeros((numcl+2,)) # acc_0,acc_1,...,acc_C-1, acc, BACC
    perf[0:numcl]=confusion_matrix.diagonal()/confusion_matrix.sum(axis=1)
    perf[numcl]=confusion_matrix.diagonal().sum()/confusion_matrix.sum(axis=1).sum()
    perf[numcl+1]=np.mean(perf[0:numcl])
    return perf,confusion_matrix
    
def change_max_num_epoch_change_learning_rate(max_num_epoch_change_learning_rate,max_num_epoch_change_rate):
    max_num_epoch_change_learning_rate= int(math.ceil(max_num_epoch_change_rate * max_num_epoch_change_learning_rate))
    return max_num_epoch_change_learning_rate    

def drange(start, stop, step):
    values=[]
    r = start
    while r <= stop:
        values.append(r)
        r += step
    return values 
    
def write_feature_weight(weights,features,lambda1s,filename):
    """
    Write the input layer weights to a file.
    Only applicable to deep feature selection.
    
    INPUTS:
    weights: numpy 2d array or matrix, 
    rows corresponding to values of lambda1,
    columns corresponding to features.
    
    features: numpy 1d array or vector, name of features.
    
    lambda1s: numpy 1d array or vector, values of lambda1.
    
    filename: string, file name to be written.
    
    OUTPUTS: 
    None.
    """
    # example:
    #weights=np.asarray([[1.1,2.2,3.4],[5.5,6.6,7.7]])
    #features=np.asarray(['f1','f2','f3'],dtype=object)
    #lambda1s=np.asarray([1.0,2.0])
    #write_feature_weight(weights,features,lambda1s,filename='test.txt')
    
    features=np.insert(features,0,'lambda')
    weights=np.asarray(weights,dtype=object)
    lambda1s=np.asanyarray(lambda1s,dtype=object)
    lambda1s.resize((lambda1s.shape[0],1))
    lambda1s_weights=np.hstack((lambda1s,weights))
    features.resize((1,features.shape[0]))
    features_lambda1s_weights=np.vstack((features,lambda1s_weights))
    np.savetxt(filename,features_lambda1s_weights,fmt='%s',delimiter='\t')

def write_feature_weight2(weights=None, features=None, lambda1s=None, accuracy=None, uniqueness=False, tol=1e-4, filename='selected_features.txt'):
    """
    Write the input layer weights and other information to a file.
    Only applicable to deep feature selection.
    
    INPUTS:
    weights: numpy 2d array or matrix, 
    rows corresponding to values of lambda1,
    columns corresponding to features.
    
    features: numpy 1d array or vector, name of features.
    
    lambda1s: numpy 1d array or vector, values of lambda1.
    
    accuracy: numpy 1d array or vector, accuracy corresponding to each lambda1.
    This parameter is optional.
    
    uniqueness: bool, indiates if only writing unique sizes of feature subsets.
    
    tol: threshold, weights below tol*w_max are considered to be zeros.
    
    filename: string, file name to be written.
    
    OUTPUTS: 
    The output file is arranged as [lambda,accuracy,num_selected,feature_subset,weights_of_feature_subset]
    """
    weights=np.asarray(weights,dtype=float)
    lambda1s=np.asarray(lambda1s,dtype=float)
    num_selected=np.zeros(len(lambda1s),dtype=int) # for each lambda, save the number of selected features
    features_selected=np.zeros(len(lambda1s),dtype=object)    
    # get the numbers of selected features
    for i in range(len(lambda1s)):
        w=weights[i]
        w_max=np.max(abs(w))
        w_min=np.min(abs(w))
        if tol*w_max<=w_min: # there is no element that is much larger: either none selected, or select all
            continue
        selected=(abs(w)>tol*w_max)
        #selected=(abs(w)>tol)
        num_selected[i]=selected.sum()
        feat_selected=features[selected]
        w_selected=w[selected]
        ind=np.argsort(abs(w_selected))
        ind=ind[::-1]
        feat_selected=feat_selected[ind]
        features_selected[i]=','.join(feat_selected)
        
    # take the first non-zeros
    if uniqueness:
        take=take_first(num_selected)
    else:
        take=np.ones(len(num_selected),dtype=bool)
    weights_take=weights[take]
    lambda1s_take=lambda1s[take]
    lambda1s_take.resize((lambda1s_take.shape[0],1))
    lambda1s_take.round(decimals=6)
    features_take=features_selected[take]
    features_take.resize((features_take.shape[0],1))
    num_take=num_selected[take]
    # if no subset is selected
    if num_take.shape[0]==0:
        return None         
    # if the last one is zero, then it means that all features are selected
    if num_take.shape[0]>1 and num_take[-1]==0 and num_take[-2]>0:
        num_take[-1]=len(features)
        features_take[-1]=','.join(features)    
    num_take.resize((num_take.shape[0],1))
    
    if accuracy is not None:
        accuracy=np.asarray(accuracy,dtype=float)
        accuracy_take=accuracy[take]
        accuracy_take.resize((accuracy_take.shape[0],1))
        accuracy_take.round(decimals=4)
        features=np.insert(features,0,['lambda','accuracy','num_selected','feature_subset'])
        features.resize((1,features.shape[0]))
        
        data=np.hstack((lambda1s_take,accuracy_take, num_take,features_take,weights_take))
        data=np.vstack((features,data))
    else:
        features=np.insert(features,0,['lambda','num_selected','feature_subset'])
        features.resize((1,features.shape[0]))
        data=np.hstack((lambda1s_take,num_take,features_take,weights_take))
        data=np.vstack((features,data))
    np.savetxt(filename,data,fmt='%s',delimiter='\t')
   
def take_first(nums):
    """
    Return the first distinct nonzeros.
    Yifeng Li in UBC.
    Aug 30, 2014.
    Example:
    nums=[0,0,0,1,2,2,2,3,4,4,5,5,5,5,6,7,7,8]
    take_first(nums)
    """
    take=np.zeros(len(nums),dtype=bool)
    if len(nums)==1:
        if nums[0]!=0:
            take[0]=True
        return take
    i=0
    while i<len(nums)-1:
        if nums[i]==0:
            i=i+1
            continue
        if i==0 and nums[i]==nums[i+1]:
            take[i]=True
        if i>0 and nums[i-1]==0:
            take[i]=True
        if i==0 and nums[i] != nums[i+1]:
            take[i]=True
            take[i+1]=True
        if nums[i] != nums[i+1]:
            take[i+1]=True
        i=i+1    
    return take

def reduce_sample_size(data,classes,times=2):
    """
    Reduce sample by to 1/times.
    
    INPUTS: 
    data: numpy 2d array or matrix, each row is a sample, the original data.
    
    classes: numpy 1d array or vector, class labels, the original labels.
    
    times: int.
    
    OUTPUTS:
    data: the reduced data.
    
    clases: the reduced classes.
    """
    data=data[range(0,data.shape[0],times)]
    classes=classes[range(0,classes.shape[0],times)]
    return data,classes
    
    