import numpy as np
import matplotlib.pyplot as plt
import math
import time



dataset = np.load('dataset.npy')
lables = np.load('lables.npy')

def sigmoid(x):
    return  1 /( 1 + (math.e)**(-1 * x))

def sigmoid_deriviate(x):
    a = sigmoid(x)
    a = np.reshape(a,(-1,1))
    b = 1 - sigmoid(x)
    b = np.reshape(b,(-1,1))
    b = np.transpose(b)
    return np.diag(np.diag(np.matmul(a,b)))


minn = np.min(dataset[:,:])
maxx = np.max(dataset[:,:])

for i in range(np.shape(dataset)[0]):
    for j in range(np.shape(dataset)[1]):
        dataset[i,j] = (dataset[i,j] - minn) / (maxx - minn) 

train_rate=0.7;
eta_e1 = 0.05;
eta_e2 = 0.03;
eta_p = 0.09;

epochs_ae = 50;
max_epoch_p = 100;

split_line_number = int(np.shape(dataset)[0] * train_rate)
x_train = dataset[:split_line_number]
x_test = dataset[split_line_number:]
y_train = lables[:split_line_number]
y_test = lables[split_line_number:]

input_dimension = np.shape(x_train)[1]

n0_neurons = input_dimension;
n1_neurons = 1000;
n2_neurons = 500;

w_e1 = np.random.uniform(low=-1,high=1,size=(n1_neurons,n0_neurons))
w_e2 = np.random.uniform(low=-1,high=1,size=(n2_neurons,n1_neurons))
w_d1 = np.random.uniform(low=-1,high=1,size=(n0_neurons,n1_neurons))
w_d2 = np.random.uniform(low=-1,high=1,size=(n1_neurons,n2_neurons))

l1_neurons=50;
l2_neurons=1;

w1 = np.random.uniform(low=-1,high=1,size=(l1_neurons,n2_neurons))
w2 = np.random.uniform(low=-1,high=1,size=(l2_neurons,l1_neurons))


#Encoder1
errr = 0
for i in range(epochs_ae):
    for j in range(np.shape(x_train)[0]):
        # Feed-Forward

        #Encoder 1
        x = np.reshape(x_train[j],(-1,1))
        net_e1 = np.matmul(w_e1, x)
        h1 = sigmoid(net_e1)
        
        #Decoder1
        net_d1 = np.matmul(w_d1, h1)
        x_hat = sigmoid(net_d1)


        # Error
        err = x - x_hat
#       errr = errr + sum(err)/1557
        # Back propagation
        
        f_driviate_d = np.diag(np.multiply(x_hat,(1-x_hat)))
        
        delta_w_d1 = eta_e1 * -1 * 1 * np.multiply(err,np.transpose(h1)) * f_driviate_d

        w_d1 = np.subtract(w_d1 , delta_w_d1)
        w_e1 = np.transpose(w_d1)

    print("AE1 : ",i)


#Encoder2
errr = 0
for i in range(epochs_ae):
    for j in range(np.shape(x_train)[0]):
        # Feed-Forward

        #Encoder 1
        x = np.reshape(x_train[j],(-1,1))
        net_e1 = np.matmul(w_e1, x)
        h1 = sigmoid(net_e1)
        
        #Encoder 2
        net_e2 = np.matmul(w_e2, h1)
        h2 = sigmoid(net_e2)
        
        #Decoder2
        net_d2 = np.matmul(w_d2, h2)
        h1_hat = sigmoid(net_d2)
        h1_hat = np.reshape(h1_hat,(-1,1))

        # Error
        err = h1 - h1_hat
   #     errr = errr + sum(err)/1000
        
        # Back propagation
        f_driviate_d = np.diag(np.multiply(h1_hat,(1-h1_hat)))
        
        delta_w_d2 = eta_e1 * -1 * 1 * np.multiply(err,np.transpose(h2)) * f_driviate_d

        w_d2 = np.subtract(w_d2 , delta_w_d2)
        w_e2 = np.transpose(w_d2)

    print("AE2 : ",i)
#    errr = 0

# MLP 2 layers
MSE_train = []
MSE_test = []
acc_train = []
acc_test = []
for i in range(max_epoch_p):

    print("")
    print("MLP Epoch: ",i+1)
    
    sqr_err_epoch_train = []
    sqr_err_epoch_test = []

    output_train = []
    output_test = []

    tp = 0
    tn = 0
    for j in range(np.shape(x_train)[0]):
        # Feed-Forward
        

        #Encoder 1
        x = np.reshape(x_train[j],(-1,1))
        net_e1 = np.matmul(w_e1,x)
        h1 = sigmoid(net_e1)
        
        #Encoder 2
        net_e2 = np.matmul(w_e2, h1)
        h2 = sigmoid(net_e2)

        #MLP 1
        net1 = np.matmul(w1,h2)
        o1 = sigmoid(net1)

        #MLP 2
        net2 = np.matmul(w2,o1)
        o2 = sigmoid(net2)
        
        if o2<=0.5:
            o2 = 0
        else:
            o2 = 1
        

        output_train.append(o2)

        # Error
        err = y_train[j] - o2
        sqr_err_epoch_train.append(err**2)
       
        if (y_train[j] == 1 and o2 == 1):
            tp = tp + 1
        elif (y_train[j] == 0 and o2 == 0):
            tn = tn + 1



        # Back propagation
        f_driviate = sigmoid_deriviate(net1)
        f_driviate2 = sigmoid_deriviate(net2)
        

        mlp_w2_f_deriviate = np.matmul(w2,f_driviate)


        mlp_w2_f_deriviate_x = np.matmul(h2,mlp_w2_f_deriviate)

        w1 = np.subtract(w1 , np.transpose((eta_p * err * -1 * f_driviate2 * mlp_w2_f_deriviate_x)))

        w2 = np.subtract(w2 , np.transpose((eta_p * err * -1 * f_driviate2 * o1)))

    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train))/np.shape(x_train)[0])
    MSE_train.append(mse_epoch_train)
#   print("Train Accuracy is: ", round((tp+tn)/len(y_train)*100,2) ,"%")
    acc_train.append(round((tp+tn)/len(y_train)*100,2))


    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for j in range(np.shape(x_test)[0]):
        # Feed-Forward

        #Encoder 1
        x = np.reshape(x_test[j],(-1,1))
        net_e1 = np.matmul(w_e1,x )
        h1 = sigmoid(net_e1)
        
        #Encoder 2
        net_e2 = np.matmul(w_e2, h1)
        h2 = sigmoid(net_e2)
        
        #MLP 1
        net1 = np.matmul(w1,h2)
        o1 = sigmoid(net1)

        #MLP 2
        net2 = np.matmul(w2,o1)
        o2 = sigmoid(net2)
        
        if o2<=0.5:
            o2 = 0
        else:
            o2 = 1
        
        output_test.append(o2)

        # Error
        err = y_test[j] - o2
        sqr_err_epoch_test.append(err ** 2)
        if (y_test[j] == 1 and o2 == 1):
            tp = tp + 1
        elif (y_test[j] == 0 and o2 == 0):
            tn = tn + 1
        elif (y_test[j] == 0 and o2 == 1):
            fp = fp + 1        
        elif (y_test[j] == 1 and o2 == 0):
            fn = fn + 1


    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test))/np.shape(x_test)[0])
    MSE_test.append(mse_epoch_test)
#    print("  TP:", tp , "  TN:",tn , "  FP:", fp , "  FN:",fn)
    acc_test.append(round((tp+tn)/len(y_test)*100,2))
print("Test Accuracy is: ", round((tp+tn)/len(y_test)*100,2) ,"%")
print("Test MSE is: ", round(mse_epoch_test,3))
