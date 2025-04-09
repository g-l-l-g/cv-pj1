import os
import json
import itertools
from datetime import datetime
from train import model_train


# 定义超参数搜索类，方法为随机搜索
class HyperParamSearch:
    def __init__(self, search_space):
        self.search_space = search_space
        self.results = {}
        self.best_model = None
        self.best_params = None
        self.best_val_acc = 0.0
        self.save_dir = f".\\hyperparameter_search_models\\{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(self.save_dir, exist_ok=True)

    def generate_combinations(self):
        keys = self.search_space.keys()
        values = (self.search_space[key] for key in keys)
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # 在save_dir目录下保存最优权重文件'best_results.json'和全部搜索结果文件'all_results.json'
    def save_results(self):
        with open(os.path.join(self.save_dir, 'best_results.json'), 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_val_acc': self.best_val_acc,
                'best_model': self.best_model
            }, f, indent=2)

        with open(os.path.join(self.save_dir, 'all_results.json'), 'w') as f:
            json.dump({
                'all_results': self.results
            }, f, indent=2)

    def run_search(self, num_trials=50):
        from random import sample
        all_combos = self.generate_combinations()
        sampled_combos = sample(all_combos, min(num_trials, len(all_combos)))

        for i, params in enumerate(sampled_combos):
            print(f"\n=== Trial {i+1}/{len(sampled_combos)} ===")
            print("Params:", json.dumps(params, indent=2))

            # 创建试验专属目录
            trial_dir = os.path.join(self.save_dir, f"trial_{i+1}")
            os.makedirs(trial_dir, exist_ok=True)

            # 保存当前试验参数到目录
            with open(os.path.join(trial_dir, 'hparams.json'), 'w') as f:
                json.dump(params, f, indent=2)

            # 调用修改后的训练函数
            val_acc, _, _, _, _ = model_train(
                save_dir=trial_dir,
                hidden_layer_size=params['hidden_layer_size'],
                act_func=params['act_func'],
                lambda_list=params['lambda_list'],
                initialize_method=params['initialize_method'],
                init_lr=params['init_lr'],
                step_size=params['step_size'],
                gamma=params['gamma'],
                batch_size=params['batch_size'],
                num_epochs=2,
                log_iters=params['log_iters'],
            )

            # 记录结果
            result = {
                'params': params,
                'val_acc': float(val_acc.item()),
                'model_path': trial_dir
            }
            self.results[f"trial_{i+1}"] = result

            # 更新最佳结果
            if val_acc > self.best_val_acc:
                self.best_val_acc = float(val_acc.item())
                self.best_params = params
                self.best_model = trial_dir

        # 保存
        self.save_results()


# ==================== 搜索空间配置 ====================
# 参数配置，具体见train.py中model_train()函数定义
search_config = {
    'hidden_layer_size': [256, 512, 1024, 2048],
    'act_func': ['ReLU', 'Tanh'],
    'lambda_list': [
        None,
        [0.01, 0.001],
    ],
    'initialize_method': ['HeInit', 'XavierInit'],
    'init_lr': [0.1, 0.05],
    'step_size': [10, 15],
    'gamma': [0.3, 0.1],
    'batch_size': [8, 32],
    'log_iters': [20, 50, 100]
}


# ==================== 执行搜索 ====================
if __name__ == "__main__":
    searcher = HyperParamSearch(search_config)
    searcher.run_search(num_trials=2)
    print(f"\nBest Params: {json.dumps(searcher.best_params, indent=2)}")
    print(f"Best Val Acc: {searcher.best_val_acc:.2%}")
