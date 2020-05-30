import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

images=[]
for f in glob.glob("D:/for ml/face rec/*.JPG") :
    a=cv.imread(f)
    v=cv.resize(a,(200,200))
    images.append(v)
    k=0xFF
    if(k==ord('q')):
        break
cv.destroyAllWindows()

y= [1,0,1,1,0,1,1,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,1,1,1,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,1,1,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1]

X=np.asarray(images)
Y=np.asarray(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.15, random_state=0)

x_test = x_test.reshape(x_test.shape[0],-1).T
x_train = x_train.reshape(x_train.shape[0],-1).T
y_test = y_test.reshape(y_test.shape[0],-1).T
y_train = y_train.reshape(y_train.shape[0],-1).T

x_test = x_test/255 
x_train = x_train/255

def sigmoid(z):
    s= (1/(1+np.exp(-z)))
    return s,z

def relu(z):
    a=np.greater(0,z)
    return a,z

def relu_back_single(self,x):
    if(x>0):
        return 1
    else:
        return 0
    
def relu_backward(self,x):
    y=(x>0)*1
    return y
    
def sigmoid_backward(self,s):
    return (s*(1-s))

def initia_para(dim):
    para={}
    L=len(dim)
    for l in range(1, L):
        para['W' + str(l)] = np.random.randn(dim[l], dim[l - 1]) * 0.01
        para['b' + str(l)] = np.zeros((dim[l], 1))
        
        
    return para

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if(activation == "sigmoid"):
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    elif(activation == "relu"):
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    A=X
    L= len(parameters)//2
    caches=[]
    for i in range(1,L):
        A_prev=A
        W=parameters['W'+str(i)]
        
        b=parameters['b'+str(i)]
        A,cache = linear_activation_forward(A_prev, W, b, activation="relu")
        caches.append(cache)
    W=parameters['W'+str(L)]
    b=parameters['b'+str(L)]
    AL,cache= linear_activation_forward(A, W, b, activation="sigmoid")
    caches.append(cache)
    
    return AL,caches

def compute_cost(AL,Y):
    m=Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ,caches):
    A_prev, W, b = caches
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T) / m
    db = (np.sum(dZ, axis=1, keepdims=True)) / m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev,dW,db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev , dW, db= linear_backward(dZ,linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev , dW, db= linear_backward(dZ,linear_cache)
        
    return dA_prev,dW,db

def L_model_backward(AL,Y,caches):
    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache=caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,activation="sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache= caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, activation="relu") 
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
        
    return grads

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters

def L_model(X,Y,dims,learning_rate=0.0075, num_itr=3000):
    costs=[]
    param= initia_para(dims)
    for i in range(0,num_itr):
        AL, caches= L_model_forward(X,param)
        cost= compute_cost(AL,Y)
        grads=L_model_backward(AL,Y,caches)
        param= update_parameters(param,grads,learning_rate)
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if i % 100 == 0:
            costs.append(cost)
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return param
    

dims = [120000, 100, 40, 15, 7,3, 1]

parameters=L_model(x_train,y_train,dims,0.0025,1000)

y_predict_train= L_model_forward(x_train, parameters)
y_predict_test= L_model_forward(x_test, parameters)
y_predict_train=y_predict_train[0]
y_predict_test=y_predict_test[0]

for i in range(y_predict_test.shape[1]):
    if(y_predict_test[0][i]>0.5):
        y_predict_test[0][i]=1
    else:
        y_predict_test[0][i]=0
        
for i in range(y_predict_train.shape[1]):
    if(y_predict_train[0][i]>0.5):
        y_predict_train[0][i]=1
    else:
        y_predict_train[0][i]=0
        
        
print("train accuracy: {} %".format(100 - np.mean(np.abs(y_predict_train - y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(y_predict_test - y_test)) * 100))


