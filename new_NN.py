import numpy as np 

def relu(x):
    return np.maximum(0, x)

def grad_relu(x):
    return (relu(x) > 0).astype(int)

def sigmoid(x):
    return 1./(1+np.exp(-x))

def grad_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def cost(y_pred, y_target):
    num_sample = y_target.shape[0]
    return -np.sum(y_target*np.log(y_pred+1e-15) + (1-y_target)*np.log(1-y_pred+1e-15)) / num_sample 

def cost_reg(y_pred, y_target, param, reg):
    num_sample = y_target.shape[0]
    data_loss = cost(y_pred, y_target)
    reg_loss = reg * (np.sum([np.sum(w[1:, :]*w[1:, :]) for w in param])) / (2*num_sample)
    total_loss = data_loss + reg_loss
    return total_loss 

def batchnorm_forward(x, gamma=1, beta=0, e=1e-8):
    # x = x[:, 1:]
    N, D = x.shape 
    #mean
    mu = 1./N * np.sum(x, axis=0)

    #subtract mean
    xmu = x - mu 

    #var
    sq = xmu ** 2
    var = 1./N * np.sum(sq, axis=0)

    #std
    std = (var + e) ** .5

    #inverse std
    invstd = 1./std

    #excute normalize
    xhat = xmu * invstd
    # print(xhat.shape)
    #out
    out = gamma * xhat + beta
    # print(out.shape)

    #store intermediate
    cache = (xhat, gamma, xmu, invstd, std, var, e)
    # print(len(cache))

    return out, cache

def batchnorm_backward(dout, cache):
    xhat, gamma, xmu, invstd, std, var, e = cache
    # print(xhat.shape)
    #dimention of output
    N, D = dout.shape

    dbeta = np.sum(dout, axis=0, keepdims=True)
    dgamma = np.sum(dout*xhat, axis=0)
    dxhat = dout * gamma

    dinvstd = np.sum(dxhat*xmu, axis=0, keepdims=True)
    dxmu1 = dxhat * invstd 

    dstd = -1./(std**2) * dinvstd

    dvar = .5 * 1./((var+e)**.5) * dstd

    dsq = 1./N * np.ones((N, D)) * dvar

    dxmu2 = 2 * xmu * dsq

    dx1 = (dxmu1+dxmu2)
    dmu = -1*np.sum(dxmu1 + dxmu2, axis=0, keepdims=True)

    dx2 = 1./N * np.ones((N, D)) * dmu 
    dx = dx1 + dx2
    # print(dx.shape)
    # print(dgamma.shape)
    # print(dbeta.shape)

    return dx, dgamma, dbeta

def conv_forward(x, w, padding=1):
    def conv_one_sample(x, w): # 1 sample
        # num_sample = x.shape[0]
        # print(x.shape)
        # print(w.shape)

    
        filter_size = w.shape[0] - 1 #number parameter each filter, not include bias
        # print(filter_size)
        pad = np.ones((x.shape[0], padding)) #padding 
        x_pad = np.concatenate((pad, x, pad), axis=1) #4x202
        # print(x_pad.shape)
        feature_dim = x_pad.shape[0] #dimension input
        x_pad = x_pad.T.flatten()#.reshape(1, -1) #1x808

        # feature_dim = x_pad.shape[1]
        x_new = [x_pad[i: i+filter_size] for i in range(0, x_pad.size - filter_size + 1, feature_dim)]
        # print(x_new.shape)
        x_new = np.array(x_new) #200x12
        # print(x_new.shape)
        x_new = np.concatenate((np.ones((x_new.shape[0], 1)), x_new), axis=1) #add ones
        # print(x_new.shape) # done processing

        conv = np.dot(x_new, w) # one column is one filter
        # print(conv.shape)
        cache = (x_new, w)
        # print(len(cache))
        # print(conv.T.shape)
        return conv.T, cache #transpose for "one row is one filter"

    outs = []
    caches = []
    for i in range(x.shape[0]):
        out, cache = conv_one_sample(x[i], w)
        outs.append(out)
        caches.append(cache)
    outs = np.array(outs)
    # print(outs.shape)
    # print(len(caches))
    # print(caches[0])
    return outs, caches

def conv_backward(dout, caches):
    
    def backward_one_sample(dout, cache):
        x, w = cache
        filter_size = w.shape[0] - 1
        len_sample = int(filter_size / 3)
        # print(x.shape)

        dconv = dout.T  # transpose lai
        # print(dconv.shape) 
        dw = np.dot(x.T, dconv)    #conv = x.w  
        # print(dw.shape) 
        dx_new = np.dot(dconv, w.T)
        # print(dx_new.shape)
        # dx = dx[:, 1: 5] # lay lai shape ban dau # error chua sua
        dx_new = dx_new[:, 1:]
        # print(dx_new.shape)
        dx1, dx2, dx3 = dx_new[:, 0:len_sample], dx_new[:, len_sample: 2*len_sample], dx_new[:, 2*len_sample:3*len_sample]
        # print(dx2.shape)
        dx1 = np.concatenate((dx2[1: , :], np.zeros((1, dx2.shape[1]))), axis=0)
        # print(dx1.shape)
        dx3 = np.concatenate((np.zeros((1, dx3.shape[1])), dx3[:-1, :]), axis=0)
        # print(dx3.shape)
        dx = dx1 + dx2 + dx3
        # print(dx.shape)

        dx = dx.T # backprop buoc flatten()
        # print(dx.shape)
        return dx, dw
        
    dxs = []
    dws = 0
    for i in range(dout.shape[0]):
        dx, dw = backward_one_sample(dout[i], caches[i])
        dws += dw
        dxs.append(dx)
    dxs = np.array(dxs)
    dws = np.array(dws)
    # print(dxs.shape)
    # print(dws.shape)
    return dxs, dws
    

def max_pooling(x, max_pool_size=(1, 2)):
    def max_pooling_one_sample(x):
        M, N = x.shape
        K, L = max_pool_size #khoang lay max

        MK = M // K # kich thuoc ma tran sau khi max_pool
        NL = N // L
        # print(M, N)
        x_new = x.reshape(MK, K, NL, L)
        # print(x_new.shape)
        out = x_new.max(axis=(1, 3))
        # print(out.shape)
        cache = (x, out)
        return out, cache

    outs = []
    caches = []
    for i in range(x.shape[0]):
        x_max, cache = max_pooling_one_sample(x[i])
        outs.append(x_max)
        caches.append(cache)
    outs = np.array(outs)
    # print(outs.shape)
    # print(len(caches))
    # print(caches[0])
    return outs, caches
    

def max_pooling_backprop(dout, caches, max_pool_size=(1, 2)):
    def max_pooling_backprop_one_sample(dout, cache):
        x, x_max = cache # x trc va sau maxpool
        K, L = max_pool_size
        mask = np.equal(x, x_max.repeat(K, axis=0).repeat(L, axis=1)).astype(int) # tim vi tri cac gia tri max
        dout = dout.repeat(K, axis=0).repeat(L, axis=1) # tinh dao ham tai nhung diem max
        dx = dout*mask 
        # print(dx.shape)
        return dx
    
    dxs = []
    for i in range(dout.shape[0]):
        dx = max_pooling_backprop_one_sample(dout[i], caches[i])
        dxs.append(dx)
    dxs = np.array(dxs)
    # print(dxs.shape)
    return dxs
    

def FC(x, w):
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((ones, x), axis=1)
    cache = (x, w)
    z = np.dot(x, w)
    out = sigmoid(z)
    return out, cache 

def FC_backward(dout, cache):
    x, w = cache
    z = np.dot(x, w)
    da = dout * grad_sigmoid(z)
    dx = np.dot(da, w.T)
    dx = dx[:, 1:]
    dw = np.dot(x.T, da)
    return dx, dw 

    





# import numpy as np
# x = np.random.randn(100, 32, 200)
# norm, cache = batchnorm_forward(x)
# dout = np.random.randn(1000, 200)
# dx, _, __ = batchnorm_backward(dout, cache)
# w = np.random.randn(97, 32)
# _, caches = conv_forward(x, w)
# print(len(caches[0]))

# dout = sx.shape)
# dout = np.random.randn(100, 32, 200)
# dx, _ = conv_backward(dout, caches)
# x = np.random.randn(100, 32, 100)
# max_pool_size = (1, 2)
# _, caches = max_pooling(x)
# print(len(caches))
# dout = np.random.randn(100, 32, 50)
# dx = max_pooling_backprop(dout, caches)

# y_train = np.random.rand(1000, 1)
# a2 = np.random.rand(y_train.shape[0], 1)
# print(cost(a2, y_train))