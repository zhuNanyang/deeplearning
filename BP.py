import copy,numpy as np
import datetime as dt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('F:/pythonNotebook/data/',one_hot=True)
trainimg=mnist.train.images
trainlabel=mnist.train.labels
testimg=mnist.test.images
testlabel=mnist.test.labels

def dropout(x, level):  
    if level < 0. or level >= 1: 
        raise Exception('Dropout level must be in interval [0, 1[.')  
    retain_prob = 1. - level  
    sample=np.random.binomial(n=1,p=retain_prob,size=x.shape)
    return sample
# The activation function of sigmoid
def sigmoid(x):
    output=1/(1+np.exp(-x))
    return output
# the activation function of sigmoid when backpropagation
def sigmoidqiudao(x):
    return x*(1-x)
# Softmax classifier
def softmax(x, y):    
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = np.exp(x)
    probs /= np.sum(probs, axis=1, keepdims=True)    
    N = x.shape[0]   
    loss = -np.mean(np.sum(y*np.log(probs)+(1-y)*np.log(1-probs),axis=1))    
    dx=y-probs    
    dx /= N
    return dx,probs,loss
# The activation of relu
def relu(x):
    out = None    
    out = ReLU(x)
    return out
def ReLU(x):
    return np.maximum(0,x)
# The activation function of relu when backpropagation
def reluqiudao(dout,X):
    dx=None
    x=X
    dx=dout
    dx[x<=0]=0
    return dx
# The activation function of tanh
def tanh(x):
    output=np.tanh(x)
    return output
# the activation function of tanh when backpropagation
def tanhqiudao(x):
    output1=1-x**2
    return output1
def ReLU(x):
    return x * (x > 0) 

def ReLU_de(x):
    return 1. * (x > 0) 
inputdim=784  # Input dimension
layer1dim=256 # Number of hidden neurons in the first layer
layer2dim=128 # Number of hidden layer neurons in the second layer
outputdim=10  # Output dimension
lr=0.01       # Learning rate

# Initialize weights and thresholds
w1 = (2*np.random.rand(inputdim,layer1dim) - 1)/5
b1 = (2*np.random.rand(1,layer1dim)-1)/5 
w2 = (2*np.random.rand(layer1dim,layer2dim) - 1)/5
b2 = (2*np.random.rand(1,layer2dim) - 1)/5
w3 = (2*np.random.rand(layer2dim,outputdim) - 1)/5
b3 = (2*np.random.rand(1,outputdim) - 1)/5
train_epoch=10 # Train epoch
batch_size=100
cost=[]
for j in range(train_epoch):
    start_everytime = dt.datetime.now() # Record the time of each run
    total_batch=int(mnist.train.num_examples/batch_size)
    for f in range(total_batch):   
       batch_xs,batch_ys=mnist.train.next_batch(batch_size) # Get training data and labels
       layer1_in=np.dot(batch_xs,w1)+b1 # Calculate value of the first hidden layer
       layer1_out=relu(layer1_in) # The first hidden layer value through activation function
       layer1_out_dropout=layer1_out*dropout(layer1_in,0.5) # The calculation result of the hidden layer of the first layer goes through the dropout calculation
       layer2_in=np.dot(layer1_out_dropout,w2)+b2 # Calculate the value of second hidden layer
       layer2_out=relu(layer2_in) # The second hidden layer value through activation function
       layer2_out_dropout=layer2_out*dropout(layer2_in,0.5) # The calculation result of the hidden layer of the second layer goes through the dropout calculation
       layer3_in=np.dot(layer2_out_dropout,w3)+b3 # Calculate value of the output layer
       dout_out,output,Loss=softmax(layer3_in,batch_ys) # Classifying the output using the softmax classifier
       # Gradient Descent Chain Backpropagation
       dout3=dout_out # Error back propagation value on output layer
       dw3=np.dot(layer2_out_dropout.T,dout3) # Get the local gradient dw3
       db3=np.sum(dout3,axis=0) # Get the local gradient db3
       dout2=np.dot(dout3,w3.T) # Error back propagation value on second layer
       
       dout2_relu=reluqiudao(dout2,layer2_in)*dropout(layer2_in,0.5) 
       dw2=np.dot(layer1_out_dropout.T,dout2_relu) # Get the local gradient dw2
       db2=np.sum(dout2_relu,axis=0) # Get the local gradient db2
       dout1=np.dot(dout2_relu,w2.T) # Error back propagation value on first layer
       #dout1_dropout=dropout(dout1,0.8)
       dout1_relu=reluqiudao(dout1,layer1_in)*dropout(layer1_in,0.5)
       dw1=np.dot(batch_xs.T,dout1_relu) # Get the local gradient dw1
       db1=np.sum(dout1_relu,axis=0) # Get the local gradient db1
       
       w3 += lr*dw3 # Update w3
       b3 += lr*db3 # Update b3
       w2 += lr*dw2 # Update w2
       b2 += lr*db2 # Update b2
       w1 += lr*dw1 # Update W1
       b1 += lr*db1 # Update b1

       loss = - np.mean(np.sum(batch_ys * np.log(output) + (1 - batch_ys) * np.log(1 - output),axis=1)) # Loss function
       cost.append(loss)
    end_everytime = dt.datetime.now()
    all_everytime = end_everytime-start_everytime
    #print(all_everytime)
    y_predict = output
    y_predict[np.arange(y_predict.shape[0]), np.argmax(y_predict, axis=1)] = 1
    accuracy = np.sum(np.argmax(y_predict, axis=1) == np.argmax(batch_ys, axis=1)) * 1.0 / batch_ys.shape[0] # Calculate accuracy of the training data
    print(accuracy)
test_num=1000
test_x,test_y = mnist.test.next_batch(test_num) # Get testing data and label
layer1_in_test=np.dot(test_x,w1)+b1 # Calculate the value of first hidden layer
layer1_out_test=relu(layer1_in_test) # The first hidden layer value through activation function
layer2_in_test=np.dot(layer1_out_test,w2)+b2 # Calculate the value of the second hidden layer
layer2_out=relu(layer2_in_test) # The second hidden layer value through activation function
layer3_in_test=np.dot(layer2_out,w3)+b3 # Calculate the value of the output layer
dout_out1,output1,Loss1=softmax(layer3_in_test,test_y) 
y_predict = output1
y_predict[np.arange(y_predict.shape[0]), np.argmax(y_predict, axis=1)] = 1
accuracy1 = np.sum(np.argmax(y_predict, axis=1) == np.argmax(test_y, axis=1)) * 1.0 / test_y.shape[0] # Calculate the accuracy of the testing data
print(accuracy1)
