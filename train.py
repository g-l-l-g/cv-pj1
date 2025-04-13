import os
import sys
import pickle
import cupy as cp
sys.path.append(r"D:\python object\computer_vision\codes_gpu\mynn")
from mynn import models, optimizer, metric, lr_scheduler, runner, op


# ==================== 数据加载模块 ====================
def load_cifar_batch(filename):
    """加载单个CIFAR-10批次文件"""
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        images = batch['data'].astype(cp.float32).reshape(-1, 3, 32, 32)
        labels = cp.array(batch['labels'], dtype=cp.int64)
        return images, labels


def load_dataset(data_dir):
    """加载完整数据集"""
    train_images, train_labels = [], []
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f'data_batch_{i}')
        images, labels = load_cifar_batch(batch_path)
        train_images.append(cp.asarray(images))
        train_labels.append(cp.asarray(labels))

    train_images = cp.concatenate(train_images, axis=0)
    train_labels = cp.concatenate(train_labels, axis=0)

    test_path = os.path.join(data_dir, 'test_batch')
    test_images, test_labels = load_cifar_batch(test_path)

    # 从训练集划分验证集（10%）
    split_idx = int(0.9 * len(train_images))
    return (
        cp.asarray(train_images[:split_idx]), cp.asarray(train_labels[:split_idx]),  # 训练集
        cp.asarray(train_images[split_idx:]), cp.asarray(train_labels[split_idx:]),  # 验证集
        cp.asarray(test_images), cp.asarray(test_labels)  # 测试集
    )


# ==================== 数据预处理 ====================
def preprocess(images):
    """数据预处理"""
    images = images / 255.0
    return cp.asarray(images.reshape(images.shape[0], -1))


# ==================== 训练流程 ====================
def model_train(
        save_dir=r".\saved_models",
        hidden_layer_size=1024,
        act_func='ReLU',
        lambda_list=None,
        initialize_method='HeInit',
        init_lr=0.5,
        step_size=10,
        gamma=0.1,
        batch_size=16,
        num_epochs=50,
        log_iters=500
        ) -> cp.ndarray:

    """
        使用多层感知机训练CIFAR-10分类模型

        Args:
            save_dir(str): 训练数据保存的地址
                - 默认地址为：'.\\saved_models'

            hidden_layer_size (int): 隐藏层维度，须满足：
                - 值范围: ≥1 的整数
                - 推荐: 2的幂次（128/256/512/1024/2048）

            act_func (str): 隐藏层激活函数，须为支持的函数名：
                - 可选值: ['ReLU', 'Logistic', 'Tanh']
                - 默认xuan: ReLU与He初始化组合

            lambda_list (Optional[List[float]]): L2正则化系数列表：
                - 若指定: 必须与网络权重层数一致（本模型为2层权重）
                - 格式示例: [0.01, 0.001]（输入-隐藏层系数，隐藏-输出层系数）

            initialize_method (str): 权重初始化方法，须与激活函数匹配：
                - 可选方法: ['HeInit', 'XavierInit', 'UniformRandomInit', 'GaussianRandomInit']
                - 搭配建议:
                    ReLU → HeInit
                    Sigmoid/Tanh → XavierInit

            init_lr (float): 学习率初始值，需根据问题调整：
                - 典型范围: 1e-4 ~ 1.0
                - 注意: 值过大可能导致梯度爆炸

            step_size (int): 学习率调整周期（阶梯衰减）：
                - 单位: epoch数
                - 操作: 每step_size个epoch应用gamma衰减

            gamma (float): 学习率衰减因子：
                - 取值范围: [0.0, 1.0]
                - 说明: 新学习率 = 原学习率 * gamma

            batch_size (int): 单次迭代样本数：
                - 约束条件: ≤ 训练集总样本数
                - 推荐值: 32/64/128（需考虑显存容量）

            num_epochs (int): 遍历训练集的次数：
                - 常规设置: 50~200

            log_iters (int): 权重更新所需的迭代次数：
                - 单位: 迭代次数
                - 范围: 正整数，且不可设置为0

        Returns:
            test_acc(cp.ndarray), 
            train_scores(List[cp.ndarray]), 
            dev_scores(List[cp.ndarray]), 
            train_loss(List[cp.ndarray]),
            dev_loss(List[cp.ndarray])
        """

    # 数据加载与预处理
    data_dir = './dataset/cifar-10-batches-py'  # 数据集路径
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(data_dir)
    X_train = preprocess(X_train)
    X_val = preprocess(X_val)
    X_test = preprocess(X_test)

    # 模型配置
    model = models.MLP(
        size_list=[3072, hidden_layer_size, 10],  # 输入层3072 -> 隐藏层2048 -> 输出层10
        act_func=act_func,
        lambda_list=lambda_list,
        initialize_method=initialize_method,
    )

    # 训练组件配置
    loss_fn = op.MultiCrossEntropyLoss(model=model)
    opt = optimizer.SGD(init_lr=init_lr, model=model)
    scheduler = lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    # 初始化训练器
    runner_ = runner.RunnerM(
        model=model,
        optimizer=opt,
        metric=metric.accuracy,
        loss_fn=loss_fn,
        batch_size=batch_size,
        scheduler=scheduler
    )

    # 启动训练
    runner_.train(
        train_set=(X_train, y_train),
        dev_set=(X_val, y_val),
        num_epochs=num_epochs,
        log_iters=log_iters,
        save_dir=save_dir
    )

    # 最终测试
    train_scores = runner_.train_scores
    dev_scores = runner_.dev_scores
    train_loss = runner_.train_loss
    dev_loss = runner_.dev_loss
    test_acc, test_loss = runner_.evaluate((X_test, y_test))
    # print(f"\nFinal Test Performance: Loss={test_loss:.4f}, Acc={test_acc:.4f}")
    return test_acc, train_scores, dev_scores, train_loss, dev_loss
