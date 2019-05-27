import numpy as np 
import matplotlib.pyplot as ply 
import pickle 
from new_NN import *


def predict(X, param):
    w1, w2, w3, w4, w5 = param
    
    conv1, cache1 = conv_forward(X, w1)
    conv1_act = relu(conv1)
    maxpool1, cache_maxpool_1 = max_pooling(conv1_act) 

    conv2, cache2 = conv_forward(maxpool1, w2)
    conv2_act = relu(conv2)
    maxpool2, cache_maxpool_2 = max_pooling(conv2_act)

    conv3, cache3 = conv_forward(maxpool2, w3) 
    conv3_act = relu(conv3)
    maxpool3, cache_maxpool_3 = max_pooling(conv3_act)
    shape_cache = maxpool3.shape
    new_input = maxpool3.reshape(maxpool3.shape[0], -1)
    new_input_nor, cache_batchnorm = batchnorm_forward(new_input)
    
    FC1, cacheFC = FC(new_input_nor, w4) 
    FC1_nor, cache2_batchnorm = batchnorm_forward(FC1)
    FC2, cacheFC2 = FC(FC1_nor, w5) 
    return FC2

with open('../data/test_data_newNN.pkl', 'rb') as f:
    test_data = pickle.load(f)
with open('../data/total_data_newNN.pkl', 'rb') as f:
    total_data = pickle.load(f)
X_train, y_train = total_data
X_test, y_test = test_data 
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# model1 = './model/_hy_param_CNN_21.0_0.7.pkl'
# model2 = './model/_hy_param_CNN_30_0.7.pkl'
# model3 = './model/_hy_param_CNN_36.0_0.7.pkl'
# model4 = './model/_hy_param_CNN_40.0_0.7.pkl'
# model5 = './model/_hy_param_CNN_40.0_0.8.pkl'
# with open('./model/enh_CNN.pkl', 'rb') as f:
#     param = pickle.load(f)
models = [model1, model2, model3, model4, model5]
X_train_new = []
X_test_new = []
for model in models:
    with open(model, 'rb') as f:
        reg, p, param = pickle.load(f)
        X_train_pred = predict(X_train, param)
        X_test_pred = predict(X_test, param)
        if X_train_new == []:
            X_train_new = X_train_pred 
        else:
            X_train_new = np.concatenate((X_train_new, X_train_pred), axis=1)
        if X_test_new == []:
            X_test_new = X_test_pred
        else:
            X_test_new = np.concatenate((X_test_new, X_test_pred), axis=1)
print(X_train_new.shape)
print(X_test_new.shape)
# print(X_train_new[-5:])
# print(y_train[-5:])

X_pred = np.mean(X_train_new, axis=1)
X_pred = (X_pred>=0.5).astype(int)
# print(X_pred[:50])
print(np.mean(X_pred == y_train.ravel()))

X_pred = np.mean(X_test_new, axis=1)
X_pred = (X_pred>=0.5).astype(int)
print(np.mean(X_pred == y_test.ravel()))
# np.save('../data/X_train_new', X_train_new)
# np.save('../data/X_test_new', X_test_new)
# np.save('../data/y_train', y_train)
# np.save('../data/y_test', y_test)




# y_pred = predict(X_test, param)
# print(y_pred.shape)
# y_pred = (y_pred>=0.5).astype(int)
# print(np.mean(y_pred==y_test))

