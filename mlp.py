# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(x, deriv=False):
    if (deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))
    
class MultilayerPerceptron:
    """
    __init__: make instance
    
        input: 1. the number of hidden layers, 
                   2. a list of the numbers of neurons in each hidden layer
                   3. bias, if true, an extra layer is added to input layer
                   
        output: classifer, usually named clf
        
        
    fit: method to train the clf
    
        input: 1. training data, in shape ... 
                   2. training label, in shape ... 
                   3. number of epoch
                
                
    score: accuracy of the clf
    
        input: 1. testing data, in shape ... 
                   2. testing label, in shape ... 

        output: 0 < score < 1


    predict_prob: make prediction of a certain data
    
        input: input data, in shape ... 

        output: 0 < probability < 1
  
    """

    # instance variables and functions 
    def __init__(self,hidden_layer=1,neurons_in_hlayers=[4],bias=False):
        self.hidden_layer_num = hidden_layer
        self.neurons_in_hlayers = neurons_in_hlayers
        self.bias = bias
        self.syns = None
        
    def __reshapeXY(self, X, Y):
        
        if len(X) == len(Y) and len(X.shape) == len(Y.shape):
            return X, Y
        
        if len(X) == len(Y) and len(X.shape) != len(Y.shape):
            new_y = np.array([[i] for i in Y])
            return X, new_y
             
    def fit(self, X, Y, epochs=60000):
        
        X_train, Y_train = self.__reshapeXY(X, Y)
        
        #print('training with error rate... ')
        
        if self.bias:
            X_train = np.array([np.append(item,[1]) for item in X_train])
            
        data_dim,  neurons_in_input = X_train.shape
        data_dim,  neurons_in_output = Y_train.shape
        
        layer_dim = self.hidden_layer_num+2
        
        
        #neurons_in_layers = None
        neurons_in_layers = []
        neurons_in_layers.append(neurons_in_input)
        neurons_in_layers.extend(self.neurons_in_hlayers)
        neurons_in_layers.append(neurons_in_output)
        
        syns = []
        # number of synapses = layer_dim - 1
        # dimension of each synapse is  given by  num of neurons in previous layer x  num of neurons in next layer 
        for i in range(layer_dim-1):
            syn = 2*np.random.random((neurons_in_layers[i],neurons_in_layers[i+1])) - 1
            syns.append(syn)
        
        
        for j in range(epochs):

            # forward prop
            layers = []
            layers.append(X_train)
            
            for k in range(layer_dim-1):
                
                # new perception is given by layer x synpase
                perception = np.dot(layers[k], syns[k])
                layers.append(sigmoid(perception))

            # backward prop
            layer_deltas = []
            
            # last layer error = label - last layer
            layer_error = Y_train - layers[-1]
            l2_error = layer_error
            
            #if (j%(epochs/100)) == 0:
            #    print(np.mean(np.abs(l2_error)))
            
            for k in range(layer_dim-1):
                
                layer_delta = layer_error * sigmoid(layers[-k-1], deriv=True)
                layer_deltas.insert(0, layer_delta)
                
                if k+2 < layer_dim:
                    
                    # generally, 
                    # previous layer error is obtained from the next layer deltas and synpases
                    layer_error = layer_delta.dot(syns[-k-1].T)

            # update synapses by gradient descent
            for k in range(layer_dim-1):
                # the 
                syns[-k-1] += layers[-k-2].T.dot(layer_deltas[-k-1])
          
        self.syns = syns
    
    def score(self,X,Y):
        
        if self.syns == None:
            print('No classifer has been trained. Use fit method')
            return
        
        X_test, Y_test = self.__reshapeXY(X, Y)
        
        if self.bias:
            X_test = np.array([np.append(item,[1]) for item in X_test])

        data_dim,  neurons_in_input = X_test.shape
        data_dim,  neurons_in_output = Y_test.shape
        layer_dim = len(self.syns) + 1
        
        if self.syns[0].shape[0] != neurons_in_input:
            print('input neurons number does not match the first synpase. Exit now ')
            return
        
        if self.syns[-1].shape[1] != neurons_in_output:
            print('output neurons number does not match the final synpase. Exit now ')
            return
        
        layer = X_test 
        for k in range(layer_dim-1):
            perception = np.dot(layer, self.syns[k])
            layer = sigmoid(perception)
                
        l2_error = Y_test - layer
        return 1- np.mean(np.abs(l2_error))
    
    def predict_proba(self, input_X):
        
        if self.syns == None:
            print('No classifer has been trained. Use fit method')
            return
        
        if self.bias:
            input_X = np.array([np.append(item,[1]) for item in input_X])
        
        data_dim,  neurons_in_input = input_X.shape
        
        layer_dim = len(self.syns) + 1
        
        layer = input_X 
        for k in range(layer_dim-1):
            perception = np.dot(layer, self.syns[k])
            layer = sigmoid(perception)
        
        return np.c_[layer.ravel(),layer.ravel()]

