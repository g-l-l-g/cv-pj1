import os
import sys
import pickle
import numpy as np
import cupy as cp
sys.path.append(r"D:\python object\computer_vision\codes_gpu\mynn")
import mynn as nn

# 加载已训练好的MLP模型
model = nn.models.MLP()

# 修改以下函数的参数可测试不同权重文件在测试集下的准确率
model.load_model(
    r'D:\python object\computer_vision\codes_gpu\hyperparameter_search_models\20250406-205637\trial_28\best_model.pkl'
)
# CIFAR-10数据集路径
data_path = r'.\dataset\cifar-10-batches-py'


def load_cifar_batch(filename):
    """加载单个CIFAR-10批次文件"""
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        # 转换数据格式
        images = batch['data'].astype(np.float32).reshape(-1, 3 * 32 * 32)  # 32x32 RGB转3072维向量
        labels = np.array(batch['labels'], dtype=np.uint8)
        return images, labels


# 加载测试集
test_images, test_labels = load_cifar_batch(
    os.path.join(data_path, 'test_batch')  # 测试集文件路径
)

# 数据预处理
test_images = cp.asarray(test_images) / 255.0  # 归一化到[0,1]
test_labels = cp.asarray(test_labels)
# 模型推理
logits = model(test_images)

# 输出测试准确率
print("Test Accuracy:", nn.metric.accuracy(logits, test_labels))
