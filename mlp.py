# -*- coding: utf-8 -*-

import numpy as np
import random

def sigmoid(x, deriv=False):
    if (deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))
    
class MultilayerPerceptron:
    """
    __init__: make instance
    
        Args: 1. the number of hidden layers, 
                   2. a list of the numbers of neurons in each hidden layer
                   3. bias, if true, an extra layer is added to input layer
                   
        Returns: classifer, usually named clf
        
        
    fit: method to train the clf
    
        input: 1. training data, in shape ... 
                   2. training label, in shape ... 
                   3. number of iterations
                
                
    score: accuracy of the clf
    
        input: 1. testing data, in shape ... 
                   2. testing label, in shape ... 

        output: 0 < score < 1


    predict_prob: make prediction of a certain data
    
        input: input data, in shape ... 

        output: 0 < probability < 1
  
    """

    # instance variables and functions 
    def __init__(self, neurons_in_hlayers=[4], learning_rate=1e-2, dropout=None):
        self.hidden_layer_num = len(neurons_in_hlayers)
        self.neurons_in_hlayers = neurons_in_hlayers
        # self.bias = bias
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.syns = None
        self.train_history = []
        self.valid_history = []
        
    def __reshapeXY(self, X, Y):
        """an ad hoc method to make X and Y of the same dimension
        really very ad hoc..
        """
        
        if len(X) == len(Y) and len(X.shape) == len(Y.shape):
            return X, Y
        
        if len(X) == len(Y) and len(X.shape) != len(Y.shape):
            new_y = np.array([[i] for i in Y])
            return X, new_y
             
    def fit(self, X, Y, train_valid_split=False, iterations=60000):
        
        data_dim, neurons_in_input = X.shape
        _, neurons_in_output = Y.shape

        X_train, Y_train = X, Y
        if train_valid_split:
            zipped = list(zip(X,Y))
            random.shuffle(zipped)
            train = list(zip(*zipped[:int(data_dim * train_valid_split)]))
            valid = list(zip(*zipped[int(data_dim * train_valid_split):]))
            X_train, Y_train = np.array(train[0]), np.array(train[1])
            X_valid, Y_valid = np.array(valid[0]), np.array(valid[1])


        # layer_dim is the total number of layer, includes input and output
        layer_dim = self.hidden_layer_num + 2
        # neurons_in_layers keep the number of neurons in each layer
        neurons_in_layers = []
        neurons_in_layers.append(neurons_in_input)
        neurons_in_layers.extend(self.neurons_in_hlayers)
        neurons_in_layers.append(neurons_in_output)

        # syns is the neural network. number of syns = layer_dim - 1
        syns = []
        # dimension of each synapse is given by num of neurons in previous layer x num of neurons in next layer 
        for i in range(layer_dim-1):
            syn = 2 * np.random.random((neurons_in_layers[i], neurons_in_layers[i+1])) - 1
            syns.append(syn)

        print('Training with training error and accuracy... ')
        for j in range(iterations):
            ###############
            # forward prop
            ###############
            layers = []
            layers.append(X_train)
            
            for k in range(layer_dim-1):
                # new perception is given by layer x synpase
                layer = sigmoid(np.dot(layers[k], syns[k]))
                if self.dropout and k+2 < layer_dim:
                    layer *= np.random.binomial([np.ones(layer.shape)], self.dropout)[0] * \
                        (1/(1 - self.dropout))
                layers.append(layer)

            ###############
            # backward prop
            ###############
            layer_deltas = []
            
            # last layer error = label - last layer
            layer_error = - (Y_train - layers[-1])
            l2_error = layer_error
            
            # printing results
            if (j%(iterations/20)) == 0:
                print('Iterations: {}'.format(j))
                train_error = np.mean(np.abs(l2_error))
                accuracy = np.mean(
                np.equal(
                    np.argmax(Y_train, axis=1),
                    np.argmax(layers[-1], axis=1)
                ))
                if train_valid_split:
                    self.syns = syns
                    valid_error, valid_acc = self.score(X_valid,Y_valid)
                    print('Train err: {}. Valid err: {}'.format(train_error, valid_error))
                    print('Train acc: {}, Valid acc: {}'.format(accuracy, valid_acc))
                    self.train_history.append((j, train_error))
                    self.valid_history.append((j, valid_error))
                else:
                    print('Train err: {}'.format(train_error))
                    print('Accuracy: {}'.format(accuracy))
                    self.train_history.append((j, train_error))
                print()
            
            for k in range(layer_dim-1):
                
                # layer_delta and layer_error dim: batch x neurons
                layer_delta = layer_error * sigmoid(layers[-k-1], deriv=True)
                layer_deltas.insert(0, layer_delta)

                if k+2 < layer_dim:           
                    # generally, 
                    # previous layer error is obtained from the next layer deltas and synpases
                    layer_error = layer_delta.dot(syns[-k-1].T)

            #####################################
            # update synapses by gradient descent
            #####################################
            for k in range(layer_dim-1):
                # update the synapes
                syns[-k-1] -= self.learning_rate * layers[-k-2].T.dot(layer_deltas[-k-1])
          
        self.syns = syns

    def show_history():

        if self.train_history:
            xs = list(list(zip(*self.train_history))[0])
            train_history = list(list(zip(*self.train_history))[1])

        if self.valid_history:
            xs = list(list(zip(*self.valid_history))[0])
            valid_history = list(list(zip(*self.valid_history))[1])
    
    def score(self,X,Y):
        
        if self.syns == None:
            print('No classifer has been trained. Use fit method')
            return
        
        X_test, Y_test = np.array(X), np.array(Y)  
        
        # old code, deprecated
        # if self.bias:
        #     X_test = np.array([np.append(item,[1]) for item in X_test])

        data_dim, neurons_in_input = X_test.shape
        data_dim, neurons_in_output = Y_test.shape
        layer_dim = len(self.syns) + 1
        
        if self.syns[0].shape[0] != neurons_in_input:
            print('input neurons number does not match the first synpase. Exit now ')
            return
        
        if self.syns[-1].shape[1] != neurons_in_output:
            print('output neurons number does not match the final synpase. Exit now ')
            return
        
        layer = X_test 
        for k in range(layer_dim-1):
            layer = sigmoid(np.dot(layer, self.syns[k]))

        accuracy = np.mean(
            np.equal(
                np.argmax(Y_test, axis=1),
                np.argmax(layer, axis=1)
            ))
                
        l2_error = Y_test - layer
        return np.mean(np.abs(l2_error)), accuracy
    
    def predict_proba(self, input_X):
        
        if self.syns == None:
            print('No classifer has been trained. Use fit method')
            return
        
        # old code, deprecated
        # if self.bias:
        #     input_X = np.array([np.append(item,[1]) for item in input_X])
        
        data_dim,  neurons_in_input = input_X.shape
        
        layer_dim = len(self.syns) + 1
        
        layer = input_X 
        for k in range(layer_dim-1):
            layer = sigmoid(np.dot(layer, self.syns[k]))
        
        return np.c_[layer.ravel(),layer.ravel()]

