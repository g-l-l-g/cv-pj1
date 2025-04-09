from abc import ABC, abstractmethod
from initializer import HeInit, GaussianRandomInit, UniformRandomInit, XavierInit
import cupy as cp


class Layer(ABC):
    """
    Abstract base class for neural network layers.

    Subclasses must implement `forward` and `backward` methods.

    Attributes:
        params (dict): Layer parameters (e.g., weights, bias).
        grads (dict): Gradients of parameters computed during backward pass.
    """
    def __init__(self) -> None:
        self.params = {}
        self.grads = {}

    @property
    def optimizable(self) -> bool:
        """Whether the layer has trainable parameters."""
        return len(self.params) > 0

    @abstractmethod
    def forward(self, X: cp.ndarray) -> cp.ndarray:
        """
        Forward pass of the layer.

        Args:
            X (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output data of shape (batch_size, output_size).
        """
        pass

    @abstractmethod
    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """
        Backward pass of the layer.

        Args:
            grad (np.ndarray): Gradient of the loss w.r.t. the layer output.

        Returns:
            np.ndarray: Gradient of the loss w.r.t. the layer input.
        """
        pass

    def zero_grad(self) -> None:
        """Reset all gradients to zero."""
        for key in self.grads:
            self.grads[key].fill(0)


# 全连接层
class Linear(Layer):
    """
    全连接层（包含权重衰减功能）
    Args:
        in_dim (int): 输入维度
        out_dim (int): 输出维度
        initialize_method (callable): 权重初始化方法（默认np.random.normal）,可选 Xavier 或 He 初始化
        weight_decay (bool): 是否启用L2正则化（默认False）
        weight_decay_lambda (float): 正则化强度系数（默认1e-8）

    Attributes:
        params (dict): 存储优化参数 {'W': ..., 'b': ...}
            W (np.ndarray): 权重矩阵，形状[in_dim, out_dim]
            b (np.ndarray): 偏置向量，形状[1, out_dim]
        grads (dict): 存储梯度 {'W': ..., 'b': ...}
        input (np.ndarray): 缓存前向传播输入用于反向计算
    """

    def __init__(self, in_dim, out_dim, initialize_method=cp.random.normal,
                 weight_decay=False, weight_decay_lambda=1e-8) -> None:

        super().__init__()

        # 参数初始化
        size = (in_dim, out_dim)
        # supported_methods = ['HeInit', 'GaussianRandomInit', 'UniformRandomInit', 'XavierInit']
        if initialize_method == 'HeInit':
            self.params['W'] = HeInit.initialize(size)
        elif initialize_method == 'GaussianRandomInit':
            self.params['W'] = GaussianRandomInit.initialize(size)
        elif initialize_method == 'UniformRandomInit':
            self.params['W'] = UniformRandomInit.initialize(size)
        elif initialize_method == 'XavierInit':
            self.params['W'] = XavierInit.initialize(size)
        else:
            self.params['W'] = cp.random.normal(size=size)

        self.params['b'] = cp.random.normal(size=(1, out_dim))

        # 将参数和梯度存入父类字典
        self.grads['W'] = cp.zeros_like(self.params['W'])
        self.grads['b'] = cp.zeros_like(self.params['b'])

        # Record the input for backward process.
        self.input = None
        # whether using weight decay
        self.weight_decay = weight_decay
        # control the intensity of weight decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> cp.ndarray:
        return self.forward(X)

    def forward(self, X) -> cp.ndarray:
        """
        执行线性变换: output = X·W + b
            Args:
                X (np.ndarray): 输入数据，形状[batch_size, in_dim]
            Returns:
                np.ndarray: 输出数据，形状[batch_size, out_dim]
        """
        self.input = X
        return cp.dot(X, self.params['W']) + self.params['b']

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """
        计算参数梯度并返回输入梯度
        Args:
            grad (np.ndarray): 来自下一层的梯度，形状[batch_size, out_dim]
        Returns:
            np.ndarray: 传递给上一层的梯度，形状[batch_size, in_dim]
        """
        # 批的数量
        batch_size = grad.shape[0]

        # 计算梯度
        self.grads['W'] = cp.dot(self.input.T, grad) / batch_size  # 平均梯度
        self.grads['b'] = cp.sum(grad, axis=0, keepdims=True) / batch_size

        # 显示调用正则化层，不执行以下代码
        '''if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W'''

        return cp.dot(grad, self.params['W'].T)


# 交叉熵损失
class MultiCrossEntropyLoss(Layer):
    """
    Cross-entropy loss with optional built-in Softmax layer (for numerical stability).
    Use `cancel_softmax()` to disable the built-in Softmax when your model already includes one.

    Args:
        model (Layer): Reference to the neural network model (for backpropagation).
    """

    def __init__(self, model=None) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True
        self.probabilities = None
        self.labels_one_hot = None
        self.batch_size = None

    def __call__(self, predicts, labels):
        return self.forward((predicts, labels))

    def forward(self, predicts_and_labels):
        """
        Args:
            predicts_and_labels: 元组包含:
                predicts (np.ndarray): 模型输出 [batch_size, num_classes]
                labels (np.ndarray): 真实标签 [batch_size, ] 或 [batch_size, num_classes]
        Returns:
            loss (float): 平均交叉熵损失
        """
        predicts, labels = predicts_and_labels
        self.batch_size = predicts.shape[0]

        # 将标签转换为one-hot编码（如果是类别索引）
        if labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1):
            num_classes = predicts.shape[1]
            self.labels_one_hot = cp.eye(num_classes)[labels.reshape(-1)]
        else:
            self.labels_one_hot = labels.copy()

        # 内置Softmax（若未取消）
        if self.has_softmax:
            shifted_logits = predicts - cp.max(predicts, axis=1, keepdims=True)
            exp = cp.exp(shifted_logits)
            self.probabilities = exp / cp.sum(exp, axis=1, keepdims=True)
        else:
            self.probabilities = predicts.copy()

        # 计算交叉熵损失
        eps = 1e-8  # 防止log(0)
        loss = -cp.sum(self.labels_one_hot * cp.log(self.probabilities + eps)) / self.batch_size
        return loss

    def backward(self, grad_output=1.0):
        """
        计算梯度并传递给模型
        """
        if self.grads is None:
            self.grads = {}

        # 梯度计算
        if self.has_softmax:
            # 当包含Softmax时，梯度为 (prob - y_true)
            grad = (self.probabilities - self.labels_one_hot) / self.batch_size
        else:
            # 当无Softmax时，直接计算交叉熵梯度 (predictions - y_true)/predictions
            # 注意：此处假设输入已经是概率分布，实际情况可能需要根据具体实现调整
            grad = (self.probabilities - self.labels_one_hot) / self.batch_size

        # 将梯度传递给模型进行反向传播
        if self.model is not None:
            self.model.backward(grad)

    def cancel_softmax(self):
        """禁用内置的Softmax层（当模型已包含Softmax时使用）"""
        self.has_softmax = False
        return self


def softmax(X):
    x_max = cp.max(X, axis=1, keepdims=True)
    x_exp = cp.exp(X - x_max)
    partition = cp.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


# L2正则化
class L2Regularization(Layer):
    """
    L2正则化层

    Args:
        linear_layer (Linear): 需要正则化的全连接层
        lambda_ (float): 正则化强度系数（默认0.01）
    """

    def __init__(self, linear_layer: 'Linear', lambda_=0.01) -> None:
        super().__init__()
        self.linear_layer = linear_layer  # 直接绑定全连接层
        self.lambda_ = lambda_  # 正则化强度

    def forward(self, loss: float) -> float:
        """
        计算带L2正则化的总损失: loss + 0.5 * λ * ||W||^2
        """
        W = self.linear_layer.params['W']
        return loss + 0.5 * self.lambda_ * cp.sum(W ** 2)

    def backward(self, grad: float = 1.0) -> None:
        """
        将L2梯度累加到全连接层的权重梯度中: grad(W) += λ * W
        """
        self.linear_layer.grads['W'] += self.lambda_ * self.linear_layer.params['W']
        return None
