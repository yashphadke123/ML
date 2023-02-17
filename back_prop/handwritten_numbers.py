#!/usr/bin/env python
import numpy as np
import pandas as pd
import math as mt
import numpy.matlib 
x = 0
weights_all = np.empty([1,784])
bias_all = np.empty([1,1])
layer_1_activation = np.empty([1,1])
layer_2_activation = np.empty([1,1])
final = np.empty([1,1])

#importing the initial pixel values and converting to a numpy array 
train_df = pd.read_csv(r'C:/Users/Yash Phadke/OneDrive/Desktop/Neural Networks/back_prop/train.csv')
test_df = pd.read_csv(r'C:/Users/Yash Phadke/OneDrive/Desktop/Neural Networks/back_prop/test.csv')
train_np = train_df.to_numpy()
test_np = test_df.to_numpy()
final_array = test_np[0, :]

#defining the sigmoid function 
def sig(x):
    return 1/(1+np.exp(-x))

#from layer 1(784) to layer 2(16)
while x < 16:
    weights_12 = numpy.matlib.randn(1,784)
    weights_all = np.concatenate((weights_all,weights_12),axis = 1)
    bias_12 = numpy.matlib.randn(1,1)
    bias_all = np.concatenate((bias_all,bias_12),axis = 1)
    linear_add = np.matmul(weights_12,final_array)
    node_layer_1_before_sig = np.add(linear_add,bias_12)
    node_layer_1_before_sig = np.float16(node_layer_1_before_sig)
    node_layer_1_after_sig = sig(node_layer_1_before_sig)
    layer_1_activation = np.concatenate((layer_1_activation,node_layer_1_after_sig))
    x = x + 1
    print 
#from layer 2(16) to layer 3(16)
while x <16 :
    weights_23 = numpy.matlib.randn(1,16)
    weights_all = np.concatenate((weights_all,weights_23),axis = 1)
    bias_23 = numpy.matlib.randn(1,1)
    bias_all = np.concatenate((bias_all,bias_23),axis = 1)
    linear_add = np.matmul(weights_23,layer_1_activation)
    node_layer_2_before_sig = np.add(linear_add,bias_23)
    node_layer_2_before_sig = np.float16(node_layer_2_before_sig)
    node_layer_2_after_sig = sig(node_layer_2_before_sig)
    layer_2_activation = np.concatenate((layer_2_activation,node_layer_2_after_sig))
    x = x + 1 
#from layer3(16) to final 
while x < 10:
    weights_3 = numpy.matlib.randn(1,784)
    weights_all = np.concatenate((weights_all,weights_3),axis = 1)
    bias_f = numpy.matlib.randn(1,1)
    bias_all = np.concatenate((bias_all,bias_f),axis = 1)
    linear_add = np.matmul(weights_12,final_array)
    node_layer_f_before_sig = np.add(linear_add,bias_f)
    node_layer_f_before_sig = np.float16(node_layer_f_before_sig)
    node_layer_f_after_sig = sig(node_layer_f_before_sig)
    final_array = np.concatenate((layer_1_activation,node_layer_1_after_sig))
    x = x + 1