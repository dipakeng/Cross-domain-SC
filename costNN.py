# -*- coding: utf-8 -*-
"""
Created on Fri May 27 12:03:15 2016

@author: hossam
"""
import numpy as np
import neurolab as nl
import time


def costNN(x,inputs,outputs,net):
    # print(x)
    trainInput=inputs
    # print(trainInput)
    
    
    trainOutput=outputs
    # print(trainOutput)
    
    numInputs=np.shape(trainInput)[1] #number of inputs
    # print("Number of inputs"+str(numInputs))
    # print("Number of inputs"+str(np.shape(trainInput)))
    
    #number of hidden neurons
    HiddenNeurons = net.layers[0].np['b'][:].shape[0]
    # print(HiddenNeurons)
   
    ######################################

    
    split1=HiddenNeurons*numInputs
    split2=split1+HiddenNeurons
    split3=split2+HiddenNeurons
    # print("split1 = " + str(split1))
    # print("split2 = " + str(split2))
    # print("split3 = " + str(split3))
    
    # input_w = 3X8 (HiddenNeurons*numInputs) 
    input_w =x[0:split1].reshape(HiddenNeurons,numInputs)
    # print(input_w)
                       
    # layer_w = 1 X 3 (HiddenNeurons)
    layer_w=x[split1:split2].reshape(1,HiddenNeurons)
 
    # input_bias = hiddenNeurons
    input_bias=x[split2:split3].reshape(1,HiddenNeurons)
    #input_bias = np.array([0.4747,-1.2475,-1.2470])
    
    # bias_2 = 1
    bias_2 =x[split3:split3+1]

    
    net.layers[0].np['w'][:] = input_w
    net.layers[1].np['w'][:] = layer_w
    net.layers[0].np['b'][:] = input_bias
    net.layers[1].np['b'][:] = bias_2
    
    
    
    pred=net.sim(trainInput).reshape(len(trainOutput))
    
    mse = ((pred - trainOutput) ** 2).mean(axis=None)

    # print(mse)
    
    
    
    return mse