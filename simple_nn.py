import numpy as np

class Layer:
    def forward(self, x): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError
    def get_params(self): return []
    def get_grads(self): return []
    def set_params(self, params): pass

class ReLU(Layer):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class Linear(Layer):
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.dW = None
        self.db = None
        self.input = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        self.dW = np.dot(self.input.T, grad_output)
        self.db = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.W.T)
        return grad_input

    def get_params(self): return [self.W, self.b]
    def get_grads(self): return [self.dW, self.db]
    def set_params(self, params):
        self.W = params[0]
        self.b = params[1]

def im2col(x, kernel_size, stride, padding):
    # x: (N, C, H, W)
    N, C, H, W = x.shape
    kh, kw = kernel_size, kernel_size
    p = padding
    
    # 填充 (使用边缘/复制填充)
    x_padded = np.pad(x, ((0,0), (0,0), (p,p), (p,p)), mode='edge')
    
    out_h = (H + 2*p - kh) // stride + 1
    out_w = (W + 2*p - kw) // stride + 1
    
    col = np.zeros((N, C, kh, kw, out_h, out_w))
    
    for y in range(kh):
        for x_k in range(kw):
            col[:, :, y, x_k, :, :] = x_padded[:, :, y:y+stride*out_h:stride, x_k:x_k+stride*out_w:stride]
            
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col, x_padded.shape

def col2im(cols, x_shape, kernel_size, stride, padding):
    N, C, H, W = x_shape
    kh, kw = kernel_size, kernel_size
    p = padding
    
    H_padded, W_padded = H + 2*p, W + 2*p
    x_padded = np.zeros((N, C, H_padded, W_padded))
    
    out_h = (H + 2*p - kh) // stride + 1
    out_w = (W + 2*p - kw) // stride + 1
    
    cols_reshaped = cols.reshape(N, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    
    for y in range(kh):
        for x_k in range(kw):
            x_padded[:, :, y:y+stride*out_h:stride, x_k:x_k+stride*out_w:stride] += cols_reshaped[:, :, y, x_k, :, :]
            
    if p > 0:
        return x_padded[:, :, p:-p, p:-p]
    return x_padded

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He 初始化
        n_in = in_channels * kernel_size * kernel_size
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(out_channels)
        
        self.dW = None
        self.db = None
        self.col = None
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        N, C, H, W = x.shape
        out_h = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        
        self.col, _ = im2col(x, self.kernel_size, self.stride, self.padding)
        W_col = self.W.reshape(self.out_channels, -1)
        
        out = np.dot(self.col, W_col.T) + self.b
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, grad_output):
        # 梯度输出: (N, out_C, out_H, out_W)
        N, out_C, out_H, out_W = grad_output.shape
        
        grad_output_reshaped = grad_output.transpose(0, 2, 3, 1).reshape(-1, out_C)
        
        self.db = np.sum(grad_output_reshaped, axis=0)
        
        W_col = self.W.reshape(self.out_channels, -1)
        self.dW = np.dot(grad_output_reshaped.T, self.col).reshape(self.W.shape)
        
        d_col = np.dot(grad_output_reshaped, W_col)
        grad_input = col2im(d_col, self.x_shape, self.kernel_size, self.stride, self.padding)
        
        return grad_input

    def get_params(self): return [self.W, self.b]
    def get_grads(self): return [self.dW, self.db]
    def set_params(self, params):
        self.W = params[0]
        self.b = params[1]

class Flatten(Layer):
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params # numpy 数组列表
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class Sequential(Layer):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def get_grads(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.get_grads())
        return grads
    
    def set_params(self, params):
        idx = 0
        for layer in self.layers:
            layer_params = layer.get_params()
            n = len(layer_params)
            if n > 0:
                layer.set_params(params[idx:idx+n])
                idx += n

class MSELoss:
    def forward(self, y_pred, y_true):
        self.diff = y_pred - y_true
        return np.mean(self.diff ** 2)

    def backward(self):
        return 2 * self.diff / self.diff.size
