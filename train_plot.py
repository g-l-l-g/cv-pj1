import os
import matplotlib.pyplot as plt
import cupy as cp
from train import model_train


def plot_metrics(train_loss, dev_loss, dev_scores, train_set_size, batch_size, log_iters, num_epochs, save_dir):
    """分别绘制三个独立图像"""
    # 转换 CuPy 数组为 NumPy
    train_loss = [cp.asnumpy(x) for x in train_loss]
    dev_loss = [cp.asnumpy(x) for x in dev_loss]
    dev_scores = [cp.asnumpy(x) for x in dev_scores]

    num_iter_per_epoch = (train_set_size + batch_size - 1) // batch_size
    records_per_epoch = (num_iter_per_epoch - 1) // log_iters + 1

    # 计算每个epoch结束对应的记录点索引
    epoch_markers = [(epoch + 1) * records_per_epoch - 1 for epoch in range(num_epochs)]
    label_positions = [epoch * records_per_epoch + records_per_epoch // 2 for epoch in range(num_epochs)]

    # 绘制训练集损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss', color='royalblue')

    for x in epoch_markers:
        if x < len(train_loss):
            plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.xticks(label_positions, [f'Epoch {e + 1}' for e in range(num_epochs)],
               rotation=45, ha='right', fontsize=8)
    plt.xlabel('Training Progress')
    plt.ylabel('Loss')
    plt.title('Train Loss with Epoch Markers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_loss.png'), dpi=300)
    plt.close()

    # 绘制验证集损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(dev_loss, label='Dev Loss', color='royalblue')

    for x in epoch_markers:
        if x < len(dev_loss):
            plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.xticks(label_positions, [f'Epoch {e + 1}' for e in range(num_epochs)],
               rotation=45, ha='right', fontsize=8)
    plt.xlabel('Training Progress')
    plt.ylabel('Loss')
    plt.title('Dev Loss with Epoch Markers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dev_loss.png'), dpi=300)
    plt.close()

    # 绘制验证集准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(dev_scores, label='Dev accuracy', color='royalblue')

    for x in epoch_markers:
        if x < len(dev_scores):
            plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.xticks(label_positions, [f'Epoch {e + 1}' for e in range(num_epochs)],
               rotation=45, ha='right', fontsize=8)
    plt.xlabel('Training Progress')
    plt.ylabel('accuracy')
    plt.title('Dev accuracy with Epoch Markers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dev_accuracy.png'), dpi=300)
    plt.close()


def main():
    # 参数配置，具体见train.py中model_train()函数定义
    train_params = {
        'save_dir': r".\test_train_plot",
        'hidden_layer_size': 512,
        'act_func': 'ReLU',
        'lambda_list': None,
        'initialize_method': 'HeInit',
        'init_lr': 0.1,
        'step_size': 9,
        'gamma': 0.1,
        'batch_size': 16,
        'num_epochs': 30,
        'log_iters': 200
    }

    # 执行训练，保存训练结果
    _, train_scores, dev_scores, train_loss, dev_loss = model_train(**train_params)
    plot_metrics(train_loss, dev_loss, dev_scores, 45000,
                 train_params["batch_size"],
                 train_params["log_iters"],
                 train_params["num_epochs"],
                 train_params["save_dir"])


if __name__ == "__main__":
    main()
