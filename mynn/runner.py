import os
import cupy as cp
from tqdm import tqdm


class RunnerM:
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.loss_fn.model = self.model   # 绑定损失函数与模型，确保反向传播

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 100)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        # 安全创建目录
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'best_model.pkl')

        best_score = 0
        best_model = self.model
        for epoch in tqdm(range(num_epochs)):
            X, y = train_set
            X = cp.asarray(X)
            y = cp.asarray(y)
            assert X.shape[0] == y.shape[0], "样本与标签数量不一致"

            # 数据洗牌
            idx = cp.random.permutation(X.shape[0])
            X, y = X[idx], y[idx]

            # 遍历所有批次
            num_iterations = int(cp.ceil(X.shape[0] / self.batch_size))
            for iteration in range(num_iterations):
                start = iteration * self.batch_size
                end = (iteration + 1) * self.batch_size
                batch_X, batch_y = X[start:end], y[start:end]

                # 前向传播与损失计算
                logits = self.model(batch_X)
                self.loss_fn(logits, batch_y)

                # 反向传播与参数更新
                self.loss_fn.backward()
                self.optimizer.step()

                # 梯度清零
                for layer in self.model.layers:
                    layer.zero_grad()

                # 评估与日志,此方法可以减少计算量，
                if iteration % log_iters == 0:
                    # 若无需绘制图像，可以将以下三行代码注释掉，减小运算量
                    train_score, train_loss = self.evaluate(train_set)
                    self.train_loss.append(train_loss)
                    self.train_scores.append(train_score)

                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_loss.append(dev_loss)
                    self.dev_scores.append(dev_score)

                    # 打印进度
                    print(f"Epoch {epoch}, Iter {iteration}: "
                          f"Train Loss={train_loss:.4f}, Dev Loss={dev_loss:.4f}, "
                          f"Train Acc={train_score:.4f}, Dev Acc={dev_score:.4f}")

                    # 以验证集上的准确率为准更新最优模型
                    if dev_score > best_score:
                        best_model = self.model
                        best_score = dev_score
                        print(f"New best model's Acc is {best_score:.4f}")

            # 学习率调整
            if self.scheduler:
                self.scheduler.step()

        # 最优模型保存
        best_model.save_model(save_path)
        print(f"best model saved with Acc {best_score:.4f}")

    # 计算模型在数据集上的准确率和损失
    def evaluate(self, dataset):
        X, y = dataset
        X = cp.asarray(X)
        y = cp.asarray(y)

        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss

