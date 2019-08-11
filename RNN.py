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
#L=20000
#trainimg=trainimg[0:L]
#trainlabel=trainlabel[0:L]

def shujuzhuanhuan(i,x,nsteps):
    x2=list()
    for j in range(100):
        x2.append(x[i+nsteps*j])
        #x2=np.array(x2)
    return x2
    
def sigmoid(x):
    output=1/(1+np.exp(-x))
    return output
def sigmoidqiudao(x):
        return x*(1-x)
def softmax(x, y):    
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = np.exp(x)
    probs /= np.sum(probs, axis=1, keepdims=True)    
    N = x.shape[0]   
    loss = -np.mean(np.sum(y*np.log(probs)+(1-y)*np.log(1-probs),axis=1))    
    dx=y-probs    
    dx /= N    

    return dx,probs
def tanh(x):
    output=np.tanh(x)
    return output
def tanhqiudao(x):
    output1=1-x**2
    return output1
    
#binary_dim=28
inputdim=28 # Input dimension
nsteps=28 # Input sequence step
hiddendim=128 # The number of hidden layer
outputdim=10 # Output dimension
lr=0.0005 # learning rate

# Initialize weights U,v,w
U=1/5*(2*np.random.random((inputdim,hiddendim))-1)
V=1/5*(2*np.random.random((hiddendim,outputdim))-1)
W=1/5*(2*np.random.random((hiddendim,hiddendim))-1)
U1=U
V1=V
W1=W
# Initialize threshold b,c
b=1/5*(2*np.random.random((1,hiddendim))-1)
c=1/5*(2*np.random.random((1,outputdim))-1)
b1=b
c1=c
train_epoch=100 # training epoch
batch_size=100
accuracy1=list()
for j in range(train_epoch):
    start_everytime = dt.datetime.now() # Record each training time
    total_batch=int(mnist.train.num_examples/batch_size)
    for f in range(total_batch):
          
        batch_xs,batch_ys=mnist.train.next_batch(batch_size) # Get training data
            #batch_xs=batch_xs/255
        x1=list()
        cost=[]
        cost1=[]
        accuracy1=list()
        layer2_deltas=list()
        layer1_values=list()
        layer1_values.append(np.zeros(hiddendim)) # Intialize the value of hidden layer
        # forward propagation
        for i in range(nsteps):
            X=np.reshape(batch_xs,[-1,inputdim])
            # Split data into different inputs
            x=shujuzhuanhuan(i,X,nsteps)
            x=np.array(x)
            #x=x[batch_size*i:batch_size*(i+1)]
            layer1=tanh(np.dot(x,U)+np.dot(layer1_values[-1],W)+b) # Get the value of hidden layer
            layer2=np.dot(layer1,V)+c # Get the value of output layer
            layer1_values.append(layer1) # Save the hidden layer values for each step
            error,out=softmax(layer2,batch_ys) # # The value of output layer through softmax classifer
            layer2_deltas.append(error) # Save the error value
            x1.append(x)
        layer1_delta2=np.zeros(hiddendim)
        # back propagation
        for i in range(nsteps-1):
            #x1=np.reshape(batch_xs,[-1,inputdim])
            #x1=x1[batch_size*(-i-1):batch_size*(-i-2)]
            layer1_now=layer1_values[-i-1] # Get the value of hidden layer at current moment from the forward propagation
            layer1_pred=layer1_values[-i-2] # Get the value of hidden layer at previous moment from the forward propagation
            layer2_delta=layer2_deltas[-1] # Get the final error value from the forward propagation
            
            # Get the error value of hidden layer
            if (i==0): 
            
                layer1_delta=(layer1_delta2.dot(W)+layer2_delta.dot(V.T))*tanhqiudao(layer1_now)

            else:
                layer1_delta=layer1_delta2.dot(W)*tanhqiudao(layer1_now)
            
            #layer1_delta=(layer1_delta2.dot(W)+layer2_delta.dot(V.T))*sigmoidqiudao(layer1_now)
            # Get the gradients of weights U,V,W
            dV = layer1_values[-1].T.dot(layer2_delta)
            dU = x1[-i-1].T.dot(layer1_delta)
            dW = layer1_pred.T.dot(layer1_delta)
            
            # Get the gradients of thresholds b,c
            db = np.sum(layer1_delta,axis=0)
            dc = np.sum(layer2_delta,axis=0)

            layer1_delta2=layer1_delta
            
            # Update the weights of U,V,W
            W = W+dW*lr
            U = U+dU*lr
            V = V+dV*lr
            # Update the thresholds of b,c 
            b = b+db*lr
            c = c+dc*lr

            loss = -np.mean(np.sum(batch_ys*np.log(out)+(1-batch_ys)*np.log(1-out),axis=1)) # Calculate loss function
            #cost.append(loss[0])
            #cost1.append(cost[-1])
        #print(cost1)

    end_everytime = dt.datetime.now()
    all_everytime = end_everytime-start_everytime
    #print(all_everytime)
    y_predict = out
    y_predict[np.arange(y_predict.shape[0]), np.argmax(y_predict, axis=1)] = 1
    accuracy = np.sum(np.argmax(y_predict, axis=1) == np.argmax(batch_ys, axis=1)) * 1.0 / batch_ys.shape[0] # Calculate accuracy value
    print(accuracy)   

#plt.plot(cost1)
#plt.show()
