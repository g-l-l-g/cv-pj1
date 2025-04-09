import os
import json
from train import model_train


def get_ith(FILE_NAME):
    try:
        with open(FILE_NAME, "r") as f:
            ith = int(f.read().strip())
    except FileNotFoundError:
        ith = 0
    ith += 1
    with open(FILE_NAME, "w") as f:
        f.write(str(ith))
    return ith


def main():
    result_directory = r".\test_train"
    file_name = os.path.join(result_directory, "counter")
    ith = get_ith(file_name)

    # 创建保存一次训练的文件夹
    train_directory = os.path.join(result_directory, f"model_{ith}")
    os.makedirs(train_directory, exist_ok=True)

    # 参数配置，具体见train.py中model_train()函数定义
    train_params = {
        'save_dir': train_directory,
        'hidden_layer_size': 8,
        'act_func': 'Logistic',
        'lambda_list': None,
        'initialize_method': 'HeInit',
        'init_lr': 0.5,
        'step_size': 2,
        'gamma': 0.1,
        'batch_size': 16,
        'num_epochs': 5,
        'log_iters': 200
    }

    # 写入参数配置
    file_name = "train_params.json"
    file_path = os.path.join(train_directory, file_name)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(train_params, file, indent=4)

    # 执行训练，保存训练结果
    model_train(**train_params)


if __name__ == "__main__":
    main()
