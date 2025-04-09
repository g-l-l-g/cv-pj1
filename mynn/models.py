import pickle
import cupy as cp
from op import Layer, Linear
from activation_function import ReLU, Logistic, Tanh


class MLP(Layer):
    """
    A model with linear layers.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, initialize_method=cp.random.normal):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func
        self.layers = []

        if size_list is not None and act_func is not None:
            for i in range(len(size_list) - 1):

                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1], initialize_method=initialize_method)

                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]

                # 激活函数添加
                layer_f = self.activation()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def activation(self):
        """激活函数识别"""
        if self.act_func == 'Logistic':
            layer_f = Logistic()
        elif self.act_func == 'ReLU':
            layer_f = ReLU()
        elif self.act_func == 'Tanh':
            layer_f = Tanh()
        else:
            raise ValueError('act_func must be either Logistic or ReLU or Tanh')
        return layer_f

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, \
            ('Model has not initialized yet. Use model.load_model to load a model or create a new model '
             'with size_list and act_func offered.')
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list_path):
        with open(param_list_path, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.layers = []

        num_linear_layers = len(self.size_list) - 1
        for i in range(num_linear_layers):
            layer_params = param_list[i + 2]
            layer = Linear(
                in_dim=self.size_list[i],
                out_dim=self.size_list[i + 1]
            )

            # 加载参数
            layer.W = layer_params['W'].copy()
            layer.b = layer_params['b'].copy()
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = layer_params['weight_decay']
            layer.weight_decay_lambda = layer_params['lambda']
            self.layers.append(layer)

            # 添加激活函数层（最后一层不添加）
            if i < num_linear_layers - 1:
                layer_f = self.activation()
                self.layers.append(layer_f)

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if isinstance(layer, Linear):
                param_list.append({
                    'W': layer.params['W'].copy(),
                    'b': layer.params['b'].copy(),
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
