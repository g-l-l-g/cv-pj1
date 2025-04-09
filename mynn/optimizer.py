from abc import abstractmethod
from op import Linear


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable and isinstance(layer, Linear):

                for key in layer.params:
                    param = layer.params[key]
                    grad = layer.grads[key]
                    if key == 'W' and layer.weight_decay:
                        grad += layer.weight_decay_lambda * param

                    layer.params[key] = param - self.init_lr * grad


"""
class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        pass
    
    def step(self):
        pass
"""
