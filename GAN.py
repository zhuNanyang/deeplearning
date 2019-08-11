import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import os
from skimage import io,data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('F:/pythonNotebook/data/',one_hot=True)

trainimg=mnist.train.images
trainlabel=mnist.train.labels
testimg=mnist.test.images
testlabel=mnist.test.labels



train_epoch=200 # Train epoch
batch_size=200 
lr = 0.00001 # learning rate
loc=0.0 # The mean of noise
scale=1.0 # The variance of noise
z_size=100 # The dimension of noise
imgsize=784 # The dimension of real image
noise_layer1dim=150 # The number of the first hidden layer for generation model
discr_layer1dim=300 # The number of the first hidden layer for discrimination model
noise_layer2dim=300 # The number of the second hidden layer for generation model 
discr_layer2dim=150 # The number of the second hidden layer for discrimination model
noise_outputdim=784 # The dimension of the fake image
discrimination_output=1 # The number of the output layer for discrimination model

# Initialize the weights and thresholds of generation model
w1_1 = (2*np.random.rand(z_size,noise_layer1dim) - 1)/5
b1_1 = (2*np.random.rand(1,noise_layer1dim)-1)/5 
w2_1 = (2*np.random.rand(noise_layer1dim,noise_layer2dim) - 1)/5
b2_1 = (2*np.random.rand(1,noise_layer2dim) - 1)/5
w3_1 = (2*np.random.rand(noise_layer2dim,noise_outputdim) - 1)/5
b3_1 = (2*np.random.rand(1,noise_outputdim) - 1)/5
# Initialize the weights,thresholds of discrimination model
w1_2 = (2*np.random.rand(imgsize,discr_layer1dim) - 1)/5
b1_2 = (2*np.random.rand(1,discr_layer1dim)-1)/5 
w2_2 = (2*np.random.rand(discr_layer1dim,discr_layer2dim) - 1)/5
b2_2 = (2*np.random.rand(1,discr_layer2dim) - 1)/5
w3_2 = (2*np.random.rand(discr_layer2dim,discrimination_output) - 1)/5
b3_2 = (2*np.random.rand(1,discrimination_output) - 1)/5
# dropout function 
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
# The activation function of sigmoid when backpropagation
def sigmoidqiudao(x):
    return x*(1-x)
# The activation function of relu
def relu(x):
    return x * (x > 0)
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
# The activation function of tanh when backpropagation
def tanhqiudao(x):
    output1=1-x**2
    return output1
# forward propagation of generation model
def generation(w1,b1,w2,b2,w3,b3,batch_noise):
    # Similar to the forward propagation of BP Neural network
    noise_input1_1 = np.dot(batch_noise,w1)+b1
    noise_input1_2 = relu(noise_input1_1)
    noise_input2_1 = np.dot(noise_input1_2,w2)+b2
    noise_input2_2 = relu(noise_input2_1)
    noise_output_1 = np.dot(noise_input2_2,w3)+b3
    noise_output_2 = tanh(noise_output_1)
    results=[noise_input1_1,noise_input1_2,noise_input2_1,noise_input2_2,noise_output_1,noise_output_2]
    g_params = [w1, b1, w2, b2, w3, b3]
    return noise_output_2,results,g_params
# Use real data and generated data as inputs for discriminant models
def discrimination(w1,b1,w2,b2,w3,b3,X1,X2):
    # Similar to the forward propagation of BP Neural network
    X1_1 = np.reshape(X1,[-1,784])
    X = np.concatenate((X1_1,X2),axis=0)
    Y1 = np.ones((X1_1.shape[0],1))
    Y2 = np.zeros((X2.shape[0],1))
    Y = np.concatenate((Y1,Y2),axis=0)
    discr_input1_1 = np.dot(X,w1)+b1
    discr_input1_2 = relu(discr_input1_1)
    discr_input1_dropout = discr_input1_2*dropout(discr_input1_2,0.5)
    discr_input2_1 = np.dot(discr_input1_dropout,w2)+b2
    discr_input2_2 = relu(discr_input2_1)
    discr_input2_dropout = discr_input2_2*dropout(discr_input2_2,0.5)
    discr_input3_1 = np.dot(discr_input2_dropout,w3)+b3
    all_input = sigmoid(discr_input3_1)
    gen_input = discr_input3_1[batch_size:]
    rea_input = discr_input3_1[:batch_size]
    gen_output = sigmoid(gen_input)
    rea_output = sigmoid(rea_input)

    results = [X,discr_input1_1,discr_input1_2,discr_input1_dropout,discr_input2_1,discr_input2_2,discr_input2_dropout,discr_input3_1]
    g_params = [w1, b1, w2, b2, w3, b3]
    return Y,rea_output,gen_output,results,g_params
# back propagation for discrimination model
def discrfanxiangchuanbo(batch_xs_2,error,results_discr,g_params_discr):
    # Get the data and parameters of forward propagation in discrimiation
    batch_x=results_discr[0]
    discr_input1_1=results_discr[1]
    discr_input1_2=results_discr[2]
    discr_input1_dropout=results_discr[3]
    discr_input2_1=results_discr[4]
    discr_input2_2=results_discr[5]
    discr_input2_dropout=results_discr[6]
    discr_input3_1=results_discr[7]
    w1 = g_params_discr[0]
    b1 = g_params_discr[1]
    w2 = g_params_discr[2]
    b2 = g_params_discr[3]
    w3 = g_params_discr[4]
    b3 = g_params_discr[5]
    # Similar to the back propagation of BP Neural network 
    dout3=error*sigmoidqiudao(discr_input3_1)
    dw3 = np.dot(discr_input2_dropout.T,dout3)
    db3 = np.sum(dout3,axis=0)

    dout2 = np.dot(dout3,w3.T)
    dout2_relu = reluqiudao(dout2,discr_input2_1)*dropout(discr_input2_1,0.5)
    dw2 = np.dot(discr_input1_dropout.T,dout2_relu)
    db2 = np.sum(dout2_relu,axis=0)
    dout1 = np.dot(dout2_relu,w2.T)

    dout1_relu = reluqiudao(dout1,discr_input1_1)*dropout(discr_input1_1,0.5)
    dw1 = np.dot(batch_x.T,dout1_relu)
    db1 = np.sum(dout1_relu,axis=0)

    return dw1,db1,dw2,db2,dw3,db3
# back propagation for generation model
def genfanxiangchuanbo(batch_noise,y_gen_output,results_gen,g_params_gen,results_discr,g_params_discr):
    # Get the data and parameters of forward propagation in generation model and discrimination
    batch_x=results_discr[0]
    discr_input1_1=results_discr[1]
    discr_input1_2=results_discr[2]
    discr_input1_dropout=results_discr[3]
    discr_input2_1=results_discr[4]
    discr_input2_2=results_discr[5]
    discr_input2_dropout=results_discr[6]
    discr_input3_1=results_discr[7]
    w1_2 = g_params_discr[0]
    b1_2 = g_params_discr[1]
    w2_2 = g_params_discr[2]
    b2_2 = g_params_discr[3]
    w3_2 = g_params_discr[4]
    b3_2 = g_params_discr[5]

    noise_input1_1=results_gen[0]
    noise_input1_2=results_gen[1]
    noise_input2_1=results_gen[2]
    noise_input2_2=results_gen[3]
    noise_output_1=results_gen[4]
    noise_output_2=results_gen[5]

    w1_1 = g_params_gen[0]
    b1_1 = g_params_gen[1]
    w2_1 = g_params_gen[2]
    b2_1 = g_params_gen[3]
    w3_1 = g_params_gen[4]
    b3_1 = g_params_gen[5]
    # Similar to the back propagation of BP Neural network
    Y = np.ones((batch_size,1))
    error = (Y - y_gen_output)/batch_size

    dout3=error*sigmoidqiudao(discr_input3_1[batch_size:])
    
    dout2 = np.dot(dout3,w3_2.T)
    dout2_relu = reluqiudao(dout2,discr_input2_1[batch_size:])*dropout(discr_input2_1[batch_size:],0.5)
    
    dout1 = np.dot(dout2_relu,w2_2.T)
    dout1_relu = reluqiudao(dout1,discr_input1_1[batch_size:])*dropout(discr_input1_1[batch_size:],0.5)

    dout_gen = np.dot(dout1_relu,w1_2.T)

    dout3_1 = dout_gen*tanhqiudao(noise_output_2)
    dw3_1 = np.dot(noise_input2_2.T,dout3_1)
    db3_1 = np.sum(dout3_1,axis=0)

    dout2_1 = np.dot(dout3_1,w3_1.T)
    dout2_1_relu = reluqiudao(dout2_1,noise_input2_1)
    dw2_1 = np.dot(noise_input1_2.T,dout2_1_relu)
    db2_1 = np.sum(dout2_1_relu,axis=0)

    dout1_1 = np.dot(dout2_1_relu,w2_1.T)
    dout1_1_relu = reluqiudao(dout1_1,noise_input1_1)

    dw1_1 = np.dot(batch_noise.T,dout1_1_relu)
    db1_1 = np.sum(dout1_1_relu)

    return dw1_1,db1_1,dw2_1,db2_1,dw3_1,db3_1

# Convert the output of vector in generation model to a picture
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):  
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], 28, 28)) + 0.5  
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]  
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)  
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)  
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)  
    for i, res in enumerate(batch_res):  
        if i >= grid_size[0] * grid_size[1]:  
            break  
        img = (res) * 255  
        img = img.astype(np.uint8)  
        row = (i // grid_size[0]) * (img_h + grid_pad)  
        col = (i % grid_size[1]) * (img_w + grid_pad)  
        img_grid[row:row + img_h, col:col + img_w] = img  
    io.imsave(fname, img_grid)

output_path = 'F:/学习/python/对抗生成网络/output2_2' # Set the file save path

for j in range(train_epoch):
    total_batch=int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size) # Get the training data and labels
        #batch_xs_1 = np.reshape(batch_xs,[-1,784])
        batch_xs_2 = 2*batch_xs.astype(np.float32)-1
        batch_noise = np.random.normal(loc,scale,size=(batch_size,z_size)).astype(np.float32) # Get the noise 
        noise_generate,results_gen,g_params_gen = generation(w1_1,b1_1,w2_1,b2_1,w3_1,b3_1,batch_noise) # Calculate generation model
        discr_label,y_rea_output,y_gen_output,results_discr,g_params_discr = discrimination(w1_2,b1_2,w2_2,b2_2,w3_2,b3_2,batch_xs_2,noise_generate) # Calculate the discrimination model

        discr_loss = -(np.log(y_rea_output)+np.log(1-y_gen_output)) # Calculate loss of discrimination model
        gen_loss = -np.log(y_gen_output) # Calculate loss of generation model

        discr_output = np.concatenate((y_rea_output,y_gen_output),axis=0)

        error = (discr_label-discr_output)/(batch_size*2) # Get the error value 
        
        dw1_2,db1_2,dw2_2,db2_2,dw3_2,db3_2 = discrfanxiangchuanbo(batch_xs_2,error,results_discr,g_params_discr) # The back propagation of the discrimination model

        if (i % 1 == 0):
            dw1_1,db1_1,dw2_1,db2_1,dw3_1,db3_1 = genfanxiangchuanbo(batch_noise,y_gen_output,results_gen,g_params_gen,results_discr,g_params_discr) # The back propagation of the generation model
            w1_1 = g_params_gen[0]
            b1_1 = g_params_gen[1]
            w2_1 = g_params_gen[2]
            b2_1 = g_params_gen[3]
            w3_1 = g_params_gen[4]
            b3_1 = g_params_gen[5]
            # Update the weights and thresholds of generation model
            w1_1 += lr*dw1_1 
            b1_1 += lr*db1_1
            w2_1 += lr*dw2_1
            b2_1 += lr*db2_1
            w3_1 += lr*dw3_1
            b3_1 += lr*db3_1
        w1_2 = g_params_discr[0]
        b1_2 = g_params_discr[1]
        w2_2 = g_params_discr[2]
        b2_2 = g_params_discr[3]
        w3_2 = g_params_discr[4]
        b3_2 = g_params_discr[5]    
        # Update the weights and thresholds of discrimination model
        w1_2 += lr*dw1_2
        b1_2 += lr*db1_2
        w2_2 += lr*dw2_2
        b2_2 += lr*db2_2
        w3_2 += lr*dw3_2
        b3_2 += lr*db3_2


    noise_generate_1,results_gen_1,g_params_gen_1 = generation(w1_1,b1_1,w2_1,b2_1,w3_1,b3_1,batch_noise) # Generate new image
    show_result(noise_generate_1,os.path.join(output_path,"sample%s.jpg" % j)) # Convert the resulting vector to a picture 
    #z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    #noise_generate_1,results_gen_1,g_params_gen_1 = generation(w1_1,b1_1,w2_1,b2_1,w3_1,b3_1,z_random_sample_val)
    #show_result(noise_generate_1, os.path.join(output_path, "random_sample%s.jpg" % j)) 
