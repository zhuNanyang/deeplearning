
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('F:/pythonNotebook/data/',one_hot=True)

numfilter1=64 # Number of convolution kernels
filtersize=3 # Size of convolution kernels
hiddendim=100 # Number of hidden layer in fully connected layer
outputdim=10 # Dimension of output layer in fully connected layer
lr=0.001 # Learning rate
# Initialize convolution kernels in convolution layer,weights and thresholds in fully connected layer 
w1 = np.random.randn(numfilter1,1,filtersize,filtersize)
b1 = np.zeros(numfilter1)
w2 = np.random.randn(numfilter1*196,hiddendim)
b2 = np.zeros((1,hiddendim))
w3 = np.random.randn(hiddendim,outputdim)
b3 = np.zeros((1,outputdim))

convparam={'stride':1,'pad':1} # Set the size of stride and padding in convolution layer
poolparam={'height':2,'width':2,'stride':2} # Set the height,weight and stride of pooling 

# Activation function in convolution layer and fully connected layer
def relu(x):
    return x * (x>0)

# Activation function in convolution layer and fullu connected layer when backpropagation
def reluqiudao(dout,X):
    dx=None
    dx=dout
    dx[X<=0]=0
    return dx

# softmax classifier
def softmax(x, y):    
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    #probs = np.exp(x)
    probs /= np.sum(probs, axis=1, keepdims=True)    
    N = x.shape[0]   
    loss = -np.mean(np.sum(y*np.log(probs)+(1-y)*np.log(1-probs),axis=1))    
    dx=y-probs    
    dx /= N    

    return dx,probs,loss  
# Calculate convolution   
def convolution(X,W,b,con):
    X1=np.reshape(X,[-1,1,28,28])
    # Get the size of stride and pad in convolution layer
    stride = con['stride']
    pad = con['pad']
    N1,C1,H1,W1 = X1.shape
    N2,C2,H2,W2 = W.shape
    # Get X2 after padding x1 in zero
    X2 = np.pad(X1,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')
    # Get the new height and weight after calculating convolution
    Heightn = 1+(H1-H2+2*pad)/stride
    Weightn = 1+(W1-W2+2*pad)/stride

    Heightn_1=int(Heightn)
    Weightn_1=int(Weightn)

    out = np.zeros((N1,N2,Heightn_1,Weightn_1))
    # Get the out of convolution calculation
    for i in range(N1):
        for j in range(N2):
            for k in range(Heightn_1):
                for f in range(Weightn_1):
                    out[i,j,k,f] = np.sum(X2[i,:,k*stride:k*stride+H2,f*stride:f*stride+W2])

    cache=(X1,W,b,convparam)
    return out,cache
# Calculate pooling layer
def pooling(X,poo):
    # Get the size of pooling 
    height = poo['height']
    width = poo['width']
    stride = poo['stride']

    N1,C1,H1,W1 = X.shape
    
    # Get the new height,weight after pooling calculation
    Heightn = 1+(H1-height)/stride
    Weightn = 1+(W1-width)/stride

    Heightn_1 = int(Heightn)
    Weightn_1 = int(Weightn)

    out = np.zeros((N1,C1,Heightn_1,Weightn_1))
    # Get the output of pooling calculation
    for i in range(N1):
        for j in range(C1):
            for k in range(Heightn_1):
                for f in range(Weightn_1):
                    X1 = X[i,j,k*stride:k*stride+height,f*stride:f*stride+width]
                    out[i,j,k,f] = np.max(X1)
    cache=(X,poo)
    return out,cache
# Calculate fully connected layer
def qianxiangchuanbo(X,W2,B2,W3,B3,N):
    # Similar to the forward propagation process of BP neural networks
    x=np.reshape(X,[-1,N*196])
    inhidden1=np.dot(x,W2)+B2
    inhidden2=relu(inhidden1)
    out1=np.dot(inhidden2,W3)+B3
    cache=(x,inhidden2,inhidden1,W2,B2,W3,B3)
    return out1,cache
def fanxiangchuanbo(dx,cache):
    # Similar to the back propagation process of BP neural networks
    x,inhidden2,inhidden1,W2,B2,W3,B3=cache
    dinhidden2=dx.dot(W3.T)
    dW3=np.dot(inhidden2.T,dx)
    dB3=np.sum(dx,axis=0,keepdims=True)
    dx1=reluqiudao(dinhidden2,inhidden1)
    dW2=np.dot(x.T,dx1)
    dB2=np.sum(dx1,axis=0,keepdims=True)
    dx2=dx1.dot(W2.T)
    return dx2,dW3,dB3,dW2,dB2
def poolingfanxiang(dx,cache): # dx is the error value in fully connected layer
    # Get input and pooling parameters for forward propagation of the pooling layer
    x,poolparam = cache 
    height=poolparam['height']
    width=poolparam['width']
    stride=poolparam['stride']
    N1,C1,H1,W1 = x.shape
    # Get the new height and weight of back propagation in pooling layer
    Heightn_1=int(1+(H1-height)/stride)
    Weightn_1=int(1+(W1-width)/stride)

    dx1 = np.reshape(dx,[N1,C1,Heightn_1,Weightn_1])
    dx2 = np.zeros_like(x)
    # Get the error value of the back propagation process to the pooled layer
    for i in range(N1):
        for j in range(C1):
            for k in range(Heightn_1):
                for f in range(Weightn_1):
                    x_1 = x[i, j, k*stride:height+k*stride, f*stride:width+f*stride]

                    m = np.max(x_1)
                    
                    dx2[i, j, k*stride:height+k*stride, f*stride:width+f*stride] = (x_1==m)*dx1[i, j, k,f]
    return dx2
def convfanxiang(dx,cache):# dx is the error value in pooling layer
    # Get input and convolutiom parameters for forward propagation of the convolution layer
    x,w,b,convparam=cache
    stride=convparam['stride']
    pad=convparam['pad']
    N1,C1,H1,W1=x.shape
    N2,C2,H2,W2=w.shape
    # Calculte padding
    x2=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')
    # Get the new height and weight of back propagation in convolution layer
    Heightn=1+(H1-H2+2*pad)/stride
    Weightn=1+(W1-W2+2*pad)/stride
    # Intialize weights and thresolds 
    dx1=np.zeros_like(x)
    dw=np.zeros_like(w)
    db=np.zeros_like(b)
    Heightn_1=int(Heightn)
    Weightn_1=int(Weightn)
    # Padding the initialized dx1
    dx2=np.pad(dx1, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    # Get the error value of the back propagation process in the convolution layer
    # Get the gradients of convolution kernels
    for i in range(N1):
        for j in range(N2):
            for k in range(Heightn_1):
                for f in range(Weightn_1):
                    window = x2[i,:,k*stride:k*stride+H2,f*stride:f*stride+W2]
                    db[j] += dx[i,j,k,f]
                    dw[j] += window*dx[i,j,k,f]
                    dx2[i,:,k*stride:k*stride+H2,f*stride:f*stride+W2] += w[j] *dx[i,j,k,f]
    dx1 = dx2[:,:,pad:pad+H1,pad:pad+W1]
    return dx1,dw,db
train_epoch=1 # train epoch
batch_size=6000 
for j in range(train_epoch):
    total_batch=int(mnist.train.num_examples/batch_size)
    for f in range(total_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size) # Get training data and labels
        out_cv,cache_cv = convolution(batch_xs,w1,b1,convparam) # Convolution calculation in convolution layer

        out_cv_1 = relu(out_cv) # The result of the convolution calculation is through the activation function

        out_pool,cache_pool = pooling(out_cv_1,poolparam) # Pooling calculation in pooling layer

        out_bp,cache_bp= qianxiangchuanbo(out_pool,w2,b2,w3,b3,numfilter1) # Calculate the fully connected layer

        dx_out,out,loss=softmax(out_bp,batch_ys) # The output of fully connected layer is through softmax classifer

        dx_bp,dw3,db3,dw2,db2=fanxiangchuanbo(dx_out,cache_bp) # The error value dx_out from softmax classifer back propagation in fully connected layer 
        
        dx_pl=poolingfanxiang(dx_bp,cache_pool) # The error value dx_bp from fully connected layer back propagation in pooling layer

        dx_pl_1=reluqiudao(dx_pl,out_cv) # The error value of dx_pl from pooling layer through activation function relu

        dx_con,dw1,db1=convfanxiang(dx_pl_1,cache_cv) # The error value dx_pl_l from pooling layer back propagation in convolution layer

        w1 += lr*dw1 # Update the convolution kernels
        w2 += lr*dw2 # Update the weights of first hidden layer in fully connected layer
        w3 += lr*dw3 # Update the weights of second hidden layer in fully connectd layer
 
        b1 += lr*db1 # Update the threshold in convolution layer 
        b2 += lr*db2 # Update the threshold in first hidden layer in fully connected layer
        b3 += lr*db3 # Update the threshold in second hidden layer in fully connected layer
    
    y_predict = out
    #y_predict[np.arange(y_predict.shape[0]), np.argmax(y_predict, axis=1)] = 1
    accuracy = np.sum(np.argmax(y_predict, axis=1) == np.argmax(batch_ys, axis=1)) * 1.0 / batch_ys.shape[0] # Calculate accuracy value

    print(accuracy)
        
