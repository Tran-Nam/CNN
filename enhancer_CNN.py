import numpy as np 
import matplotlib.pyplot as plt 
from new_NN import *
import pickle 
from sklearn.model_selection import train_test_split

with open('../data/data_newNN.pkl', 'rb') as f:
    data = pickle.load(f)
# with open('../data/test_data_newNN.pkl', 'rb') as f:
#     test_data = pickle.load(f)

X_train, X_val, y_train, y_val = data 
# X = data['X']
# zeros = np.zeros((4, 1))
# for i in range(X.shape[0]):
#     print(zeros.shape)
#     print(X[i].shape)
#     X[i] = np.concatenate((zeros, X[i]), axis=1)
#     print(X[i].shape)
# y = data['y']
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# X_test, y_test = test_data
print(X_train[0, :, :5])
# print(X_test[0, :, :5])
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
# print(y_train[:5])
# print(y_val[:5])

NUM_FILTER = 32
FILTER_DIM_1 = 13
FILTER_DIM_2 = 97
FILTER_DIM_3 = 97
NUM_UNIT = 128
NUM_FEATURE = 800
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
ITE = 100



def forward(X, param, p):
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
    mask1 = (np.random.rand(*FC1.shape) < p) / p 
    FC1 *= mask1 #drop
    FC1_nor, cache2_batchnorm = batchnorm_forward(FC1)
    
    FC2, cacheFC2 = FC(FC1_nor, w5)
    cache = (conv1, cache1, conv1_act, maxpool1, cache_maxpool_1, conv2, cache2, conv2_act, maxpool2, cache_maxpool_2, \
            conv3, cache3, conv3_act, maxpool3, cache_maxpool_3, shape_cache, new_input, new_input_nor, cache_batchnorm, FC1, cacheFC, \
            FC1_nor, cache2_batchnorm, FC2, cacheFC2)
    return cache

def backward(y, cache):
    num_train = y.shape[0]
    conv1, cache1, conv1_act, maxpool1, cache_maxpool_1, conv2, cache2, conv2_act, maxpool2, cache_maxpool_2, \
    conv3, cache3, conv3_act, maxpool3, cache_maxpool_3, shape_cache, new_input, new_input_nor, cache_batchnorm, FC1, cacheFC, \
    FC1_nor, cache2_batchnorm, FC2, cacheFC2 = cache

    dout = (FC2 - y) / num_train
    dFC2, dw5 = FC_backward(dout, cacheFC2)
    
    dFC1_nor, _, __ = batchnorm_backward(dFC2, cache2_batchnorm)
    dFC1, dw4 = FC_backward(dFC1_nor, cacheFC)

    dnew_input_nor, _, __ = batchnorm_backward(dFC1, cache_batchnorm)
    dnew_input = dnew_input_nor.reshape(shape_cache)
   
    dmaxpool3 = max_pooling_backprop(dnew_input, cache_maxpool_3) 
    dconv3_act = dmaxpool3 * grad_relu(conv3)
    dconv3, dw3 = conv_backward(dconv3_act, cache3)
  
    dmaxpool2 = max_pooling_backprop(dconv3, cache_maxpool_2) 
    dconv2_act = dmaxpool2 * grad_relu(conv2)
    dconv2, dw2 = conv_backward(dconv2_act, cache2)
   
    dmaxpool1 = max_pooling_backprop(dconv2, cache_maxpool_1) 
    dconv1_act = dmaxpool1 * grad_relu(conv1)
    dconv1, dw1 = conv_backward(dconv1_act, cache1)
   
    grad = (dw1, dw2, dw3, dw4, dw5)
    return grad

def update(lr, reg, num_train, param, grad):
    w1, w2, w3, w4, w5 = param 
    # print(len(grad))
    dw1, dw2, dw3, dw4, dw5 = grad
    dw1 += reg*dw1/num_train 
    dw2 += reg*dw2/num_train 
    dw3 += reg*dw3/num_train 
    dw4 += reg*dw4/num_train 
    dw5 += reg*dw5/num_train 

    #update
    w1 -= lr*dw1
    w2 -= lr*dw2
    w3 -= lr*dw3
    w4 -= lr*dw4
    w5 -= lr*dw5

    param = [w1, w2, w3, w4, w5]
    return param 


def train(X_train, y_train, X_val, y_val, reg, p, lr=LEARNING_RATE, ite=ITE, batch_size=BATCH_SIZE):

    #initializer
    w1 = np.random.randn(FILTER_DIM_1, NUM_FILTER) * 0.01
    w2 = np.random.randn(FILTER_DIM_2, NUM_FILTER) * 0.01
    w3 = np.random.randn(FILTER_DIM_3, NUM_FILTER) * 0.01
    w4 = np.random.randn(NUM_FEATURE + 1, NUM_UNIT) * (2/(NUM_FEATURE+NUM_UNIT))**0.5
    w5 = np.random.randn(NUM_UNIT + 1, 1) * (2/(NUM_UNIT))**0.5
    param = [[w1, w2, w3, w4, w5]]
    loss_train = []
    loss_val = []
    # param = []
    #lr = LEARNING_RATE
    print('Reg = {}, Drop = {}...'.format(reg, p))
    #minibatch GD
    for it in range(ite):

        result_train = predict(X_train, param[-1])
        result_val = predict(X_val, param[-1])
        loss_train.append(cost(result_train, y_train))
        loss_val.append(cost(result_val, y_val))        

        if it%10 == 0: 
        #     result_train = predict(X_train, param)
        #     result_val = predict(X_val, param)
        #     loss_train.append(cost(result_train, y_train))
        #     loss_val.append(cost(result_val, y_val))
            print(cost(result_train, y_train))
            print('Epoch : {0} \tLoss {1}'.format(it, cost_reg(result_train, y_train, param[-1], reg)))
        
        if cost(result_train, y_train) < .8:
            # lr = (1/(1+1*it))*LEARNING_RATE
            lr = 0.9**it * LEARNING_RATE
            # print(lr)

        ids = np.random.permutation(range(X_train.shape[0]))
        for batch_id in range(int(len(ids) / batch_size)):
            param_next = param[-1]
            start_id = batch_id * batch_size
            end_id = (batch_id+1) * batch_size
            if end_id + batch_size > len(ids):
                end_id = len(ids)
            batch_X = X_train[ids[start_id: end_id]]
            batch_y = y_train[ids[start_id: end_id]]
            num_train = batch_X.shape[0]
   
            cache = forward(batch_X, param[-1], p)
            # cache = forward(batch_X, param)          
            grad = backward(batch_y, cache)
            # param = update(lr, reg, num_train, param, grad)
            param_next = update(lr, reg, num_train, param_next, grad)

        param.append(param_next)
  

    opt_position = np.argmin(loss_val)
    print('Loss val min at epoch {}'.format(opt_position))
    print('Loss_val: {}'.format(np.min(loss_val)))
    param_opt = param[opt_position + 1]

    fig = plt.figure(figsize=(10, 10))
    xplot = range(0, ite)
    plt.plot(xplot, loss_train, label='Loss train')
    plt.plot(xplot, loss_val, label='Loss val')
    plt.legend()
    plt.title('Reg: {0} _ Drop: {1}'.format(reg, p))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    # plt.savefig('./new_NN/_reg_{0}_drop_{1}.png'.format(reg, p))
    plt.show()

    return param_opt, loss_train, loss_val


#LEARNING_RATE = 1e-4
# lr = 1
regs = [.1, .3, 1, 3, 10, 30, 100]

reg = 1
ps = [.2, .3, .4, .5, .6, .7, .8, .9]

p = 1 #dropout

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




param, loss_train, loss_val = train(X_train, y_train, X_val, y_val, reg=reg, p=p, batch_size=32)
out = predict(X_train, param)
out = (out>=0.5).astype(int)
out_ = predict(X_val, param)
out_ = (out_>=0.5).astype(int)

print(out[:5])
print(y_train[:5])
# print(out[:5])
print(np.mean(out == y_train))
print(np.mean(out_ == y_val))

# batches = [8, 16, 32]
# min_loss = 1
# for batch in batches:
#     print('batch = {}'.format(batch))
#     param, loss_train, loss_val = train(X_train, y_train, X_val, y_val, reg=reg, p=p, batch_size=batch)
#     loss_val_min = np.min(loss_val)
#     if loss_val_min < min_loss:
#         min_loss = loss_val_min
#         batch_opt = batch 

# print("Best batch is: {}".format(batch_opt))

from collections import defaultdict
model = defaultdict(int)
#find best reg
min_loss = 1
# min_loss_vals = []
# params = []
for reg in regs:
    param, loss_train, loss_val = train(X_train, y_train, X_val, y_val, reg=reg, p=p)
    loss_val_min = np.min(loss_val)
    model[loss_val_min] = [reg, p, param]
    # params.append(param)
    # min_loss_vals.append(loss_val_min)
    if loss_val_min < min_loss:
        min_loss = loss_val_min
        reg_opt = reg 
print('Reg_opt: {}'.format(reg_opt))
print(model)

for reg in np.linspace(max(0, reg_opt-10), reg_opt+10, 21):
    param, loss_train, loss_val = train(X_train, y_train, X_val, y_val, reg=reg, p=p)
    loss_val_min = np.min(loss_val)
    model[loss_val_min] = [reg, p, param]
    if loss_val_min < min_loss:
        min_loss = loss_val_min
        best_reg = reg
print('Best_reg: {}'.format(best_reg))

#find best drop
min_loss = 1
for p in ps:
    param, loss_train, loss_val = train(X_train, y_train, X_val, y_val, reg=best_reg, p=p)
    loss_val_min = np.min(loss_val)
    model[loss_val_min] = [reg, p, param]
    if loss_val_min < min_loss:
        min_loss = loss_val_min
        best_p = p 
print('best_p: {}'.format(best_p))

for p in np.linspace(p_opt-0.1, p_opt+0.1, 11):
    param, loss_train, loss_val = train(X_train, y_train, X_val, y_val, reg=best_reg, p=p)
    loss_val_min = np.min(loss_val)
    if loss_val_min < min_loss:
        min_loss = loss_val_min
        best_p = p
        best_param = param
print('best_p: {}'.format(best_p))

hy_param = [best_reg, best_p]
print('Best_reg: {0}\nBest_p: {1}'.format(best_reg, best_p))

# with open('./model/param_CNN.pkl', 'wb') as f:
#     pickle.dump(hy_param, f)
# with open('./model/enh_CNN.pkl', 'wb') as f:
#     pickle.dump(best_param, f)



# keys = sorted(model.keys())
# X_new = []
# X_total = np.concatenate((X_train, X_val), axis=0)
# y_total = np.concatenate((y_train, y_val), axis=0)
# for key in keys[:5]:
#     hy_param = model[key]
#     reg, p, param = hy_param
#     with open('./model/_hy_param_CNN_{0}_{1}.pkl'.format(reg, p), 'wb') as f:
#         pickle.dump(hy_param, f)
#     x_pred = predict(X_total, param)
#     if X_new == []:
#         X_new = x_pred 
#     else:
#         X_new = np.concatenate((X_new, x_pred), axis=1)

# print(X_new.shape)
# data = [X_new, y_total]
# np.save('../data/data_new.npy', data)


# out = predict(X_train, param)
# out = (out>=0.5).astype(int)
# print(np.mean(out == y_train))








# def train(X, y, lr, ite, batch_size=BATCH_SIZE):
#     w1 = np.random.randn(FILTER_DIM_1, NUM_FILTER)
#     w2 = np.random.randn(FILTER_DIM_2, NUM_FILTER)
#     w3 = np.random.randn(FILTER_DIM_3, NUM_FILTER)
#     w4 = np.random.randn(NUM_FEATURE + 1, NUM_UNIT)
#     w5 = np.random.randn(NUM_UNIT + 1, 1)
 
#     for it in range(ite):
#         if it%10 == 0: 
#             result = predict(X, w1, w2, w3, w4, w5)
#             print('Ite : {0} \tLoss {1}'.format(it, cost(result, y)))

#         ids = np.random.permutation(range(X.shape[0]))
#         for batch_id in range(int(len(ids) / batch_size)):
#             start_id = batch_id * batch_size
#             end_id = (batch_id+1) * batch_size
#             if end_id + batch_size > len(ids):
#                 end_id = len(ids)
#             batch_X = X[ids[start_id: end_id]]
#             batch_y = y[ids[start_id: end_id]]
#             # print("Batch {0}\t{1}\t{2}".format(batch_id, batch_X.shape, batch_y.shape))

#             #forward
#             conv1, cache1 = conv_forward(batch_X, w1)
#             conv1_act = relu(conv1)
#             # print(conv1_act.shape)
#             maxpool1, cache_maxpool_1 = max_pooling(conv1_act) 
#             # print(maxpool1.shape)

#             conv2, cache2 = conv_forward(maxpool1, w2)
#             # print(conv2.shape)
#             conv2_act = relu(conv2)
#             # print(conv2_act.shape)
#             maxpool2, cache_maxpool_2 = max_pooling(conv2_act)
#             # print(maxpool2.shape)

#             conv3, cache3 = conv_forward(maxpool2, w3)
#             # print(conv3.shape)
#             conv3_act = relu(conv3)
#             # print(conv3_act.shape)
#             maxpool3, cache_maxpool_3 = max_pooling(conv3_act)
#             # print(maxpool3.shape)

#             shape_cache = maxpool3.shape
#             new_input = maxpool3.reshape(maxpool3.shape[0], -1)
#             # print(new_input.shape)

#             new_input_nor, cache_batchnorm = batchnorm_forward(new_input)
#             # print(new_input_nor.shape)


            
#             # print(w4.shape)
#             FC1, cacheFC = FC(new_input_nor, w4)
#             # print(FC1.shape)
#             FC1_nor, cache2_batchnorm = batchnorm_forward(FC1)
#             # print(FC1_nor.shape)

#             FC2, cacheFC2 = FC(FC1_nor, w5)
#             # print(FC2.shape)
#             # if it%10 == 0: 
#             #     print('Iter: {0}\tCost: {1}'.format(it, cost(FC2, batch_y)))
#             #     result = (FC2 > 0.5).astype(int)
#             #     print('Acc: {}'.format(np.mean(result == batch_y)))


#             #backward
#             dout = (FC2 - batch_y) / batch_y.shape[0]
#             # print(dout.shape)
#             dFC2, dw5 = FC_backward(dout, cacheFC2)
#             # print(dFC2.shape)
#             # print(dw5.shape)


#             dFC1_nor, _, __ = batchnorm_backward(dFC2, cache2_batchnorm)
#             # print(dFC1_nor.shape)

#             dFC1, dw4 = FC_backward(dFC1_nor, cacheFC)
#             # print(dFC1.shape)


#             dnew_input_nor, _, __ = batchnorm_backward(dFC1, cache_batchnorm)
#             # print(dnew_input_nor.shape)

#             dnew_input = dnew_input_nor.reshape(shape_cache)
#             # print(dnew_input.shape)

#             dmaxpool3 = max_pooling_backprop(dnew_input, cache_maxpool_3) 
#             # print(dmaxpool3.shape)
#             dconv3_act = dmaxpool3 * grad_relu(conv3)
#             # print(dconv3_act.shape)
#             dconv3, dw3 = conv_backward(dconv3_act, cache3)
#             # print(dconv3.shape)
#             # print(dw3.shape)


#             #error conv_backward
#             dmaxpool2 = max_pooling_backprop(dconv3, cache_maxpool_2) 
#             # print(dmaxpool2.shape)
#             dconv2_act = dmaxpool2 * grad_relu(conv2)
#             # print(dconv2_act.shape)
#             dconv2, dw2 = conv_backward(dconv2_act, cache2)
#             # print(dconv2.shape)
#             # print(dw2.shape)

#             dmaxpool1 = max_pooling_backprop(dconv2, cache_maxpool_1) 
#             # print(dmaxpool1.shape)
#             dconv1_act = dmaxpool1 * grad_relu(conv1)
#             # print(dconv1_act.shape)
#             dconv1, dw1 = conv_backward(dconv1_act, cache1)
#             # print(dconv1.shape)
#             # print(dw1.shape)

#             #update
#             w1 -= lr*dw1
#             w2 -= lr*dw2
#             w3 -= lr*dw3
#             w4 -= lr*dw4
#             w5 -= lr*dw5

#         # if it%10 == 0: 
#         #     result = predict(X, w1, w2, w3, w4, w5)
#         #     print('Ite : {0} \tLoss {1}'.format(it, cost(result, y_train)))
        
#         # if it%10 == 0: 
#         #     print('Iter: {0}\tCost: {1}'.format(it, cost(FC2, y_train)))
#         #     result = (FC2 > 0.5).astype(int)
#         #     print('Acc: {}'.format(np.mean(result == y_train)))
#     return w1, w2, w3, w4, w5

# def predict(X, w1, w2, w3, w4, w5):
#     #forward
#     conv1, cache1 = conv_forward(X, w1)
#     conv1_act = relu(conv1)
#     # print(conv1_act.shape)
#     maxpool1, cache_maxpool_1 = max_pooling(conv1_act) 
#     # print(maxpool1.shape)

#     conv2, cache2 = conv_forward(maxpool1, w2)
#     # print(conv2.shape)
#     conv2_act = relu(conv2)
#     # print(conv2_act.shape)
#     maxpool2, cache_maxpool_2 = max_pooling(conv2_act)
#     # print(maxpool2.shape)

#     conv3, cache3 = conv_forward(maxpool2, w3)
#     # print(conv3.shape)
#     conv3_act = relu(conv3)
#     # print(conv3_act.shape)
#     maxpool3, cache_maxpool_3 = max_pooling(conv3_act)
#     # print(maxpool3.shape)

#     shape_cache = maxpool3.shape
#     new_input = maxpool3.reshape(maxpool3.shape[0], -1)
#     # print(new_input.shape)

#     new_input_nor, cache_batchnorm = batchnorm_forward(new_input)
#     # print(new_input_nor.shape)


    
#     # print(w4.shape)
#     FC1, cacheFC = FC(new_input_nor, w4)
#     # print(FC1.shape)
#     FC1_nor, cache2_batchnorm = batchnorm_forward(FC1)
#     # print(FC1_nor.shape)

#     FC2, cacheFC2 = FC(FC1_nor, w5)
#     return FC2


# def predict(X, param):
#     cache = forward(X, param)
#     result = cache[-2]
#     return result

# param = train(X_train, y_train, lr, ite)
# out = predict(X_train, param)
# out_ = (out>=0.5).astype(int)
# # print(out[:10])
# print(y_train[:5])
# print(out_[:5])
# print(np.mean(out_ == y_train))


    

