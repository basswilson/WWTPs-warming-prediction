import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as func
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Linear_ANN(nn.Module):
    def __init__(self, input_features, output_features, prior_var=1.):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.randn(output_features, input_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_features))

    def forward(self, input):
        return func.linear(input, self.weight, self.bias)


class Neural3network(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim, p=0):
        super(Neural3network, self).__init__()
        self.layer1 = Linear_ANN(in_dim, n_hidden_1)
        self.layer2 = Linear_ANN(n_hidden_1, out_dim)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.layer1(x))
        x = func.relu(x)
        x = torch.sigmoid(self.layer2(x))
        return x


class MyDataset(Dataset):
    def __init__(self, col=1):
        data1 = np.loadtxt('/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/GLV 数据集/E.csv', delimiter=',',
                           skiprows=1, usecols=range(1, 9), dtype=np.float32)
        # ENV8 扰动 + 5
        data1[:, 0] += 5

        data2 = np.loadtxt('/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/GLV 数据集/人工数据集.csv',
                           delimiter=',', skiprows=1, usecols=col, dtype=np.float32)
        self.data2_min = data2.min(axis=0)
        self.data2_max = data2.max(axis=0)
        data2_normed = (data2 - self.data2_min) / (self.data2_max - self.data2_min + 1e-12)

        state = np.random.get_state()
        indices = np.arange(data1.shape[0])
        np.random.shuffle(data1)
        np.random.set_state(state)
        np.random.shuffle(data2_normed)
        np.random.shuffle(indices)

        self.features = torch.from_numpy(data1)
        self.targets = torch.reshape(torch.from_numpy(data2_normed), (800, 1))
        self.length = data1.shape[0]
        self.indices = torch.from_numpy(indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.indices[idx]


def get_testdata(col=1):
    data = MyDataset(col)
    return data.features, data.targets, data.indices, data.data2_min, data.data2_max


# 初始化结果存储
os.chdir('/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/Before Perturbed/GLV/')
r2_summary = []

# 所有列合并图准备
all_true_values = []
all_predicted_values = []
all_col_labels = []

# 主处理循环
for col in range(1, 101):  # 你可以根据需要调整列范围
    print(f'\n{"=" * 40}')
    print(f'Processing column {col}')

    col_para = np.ones((0, 15))
    col_test = np.ones((0, 15))

    for seed in range(0, 5):
        try:
            model_path = f'{col}col-4fold-seed{seed}-10000ep-78bS-0.00001lr-0.01wd-drop0.2_train_network.pth'
            net = torch.load(model_path, map_location='cpu', weights_only=False)


            weight_H_To_O = np.diag(net.layer2.weight.data.numpy()[-1])
            weight_I_To_H = net.layer1.weight.data.numpy()
            weight_final = np.dot(weight_H_To_O, weight_I_To_H)
            abs_weight_final = abs(weight_final)
            denominator = abs_weight_final.sum(axis=1, keepdims=True) + 1e-12
            weightForH = abs_weight_final / denominator
            weightForH[np.isnan(weightForH)] = 0
            Sum_input = weightForH.sum(axis=0)
            RI = Sum_input / Sum_input.sum()
            para_bias = np.dot(net.layer2.weight.data.numpy(),
                               net.layer1.bias.data.numpy()) + net.layer2.bias.data.numpy()

            seed_torch(seed)
            x_test, y_test, indices, y_min, y_max = get_testdata(col)
            test_len = x_test.shape[0]

            col_result = np.ones((test_len, 1)) * col
            seed_result = np.ones((test_len, 1)) * seed
            wd_result = np.ones((test_len, 1)) * 0.01
            drop_P_result = np.ones((test_len, 1)) * 0

            net.eval()
            with torch.no_grad():
                Test_pred = net(x_test)
                Test_pred_ = Test_pred.data.numpy()
                y_test_ = y_test.numpy()

            Test_pred_denorm = Test_pred_ * (y_max - y_min + 1e-12) + y_min
            y_test_denorm = y_test_ * (y_max - y_min + 1e-12) + y_min

            each_test = np.c_[indices, col_result, seed_result, wd_result, drop_P_result,
                              x_test.numpy(), y_test_denorm, Test_pred_denorm]
            col_test = np.r_[col_test, each_test]

            each_para = np.c_[col, seed, 0.01, 0,
                              mean_squared_error(y_test_denorm, Test_pred_denorm),
                              r2_score(y_test_denorm, Test_pred_denorm),
                              RI.reshape(1, 8), para_bias]
            col_para = np.r_[col_para, each_para]

            # 收集所有点
            all_true_values.extend(y_test_denorm.flatten())
            all_predicted_values.extend(Test_pred_denorm.flatten())
            all_col_labels.extend([col] * test_len)

        except FileNotFoundError:
            print(f'  × Model not found for seed {seed}')
            continue
        except Exception as e:
            print(f'  ! Error processing seed {seed}: {str(e)}')
            continue

    results_dir = '/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/After Perturbed/GLV/analysis/individual_cols/'

    if col_test.shape[0] > 0:
        np.savetxt(f'{results_dir}Test_log_col{col}.txt', col_test, fmt='%.4e', delimiter='\t')
        np.savetxt(f'{results_dir}Test_and_parameters_col{col}.csv', col_para, fmt='%.4e', delimiter=',')

        true_values = col_test[:, -2]
        predicted_values = col_test[:, -1]
        r2 = r2_score(true_values, predicted_values)
        r2_summary.append({'Column': col, 'R2_Score': r2})
    else:
        print(f'  ! No valid data for column {col}')
        r2_summary.append({'Column': col, 'R2_Score': np.nan})

# --- 绘制合并后的图 ---
# --- 绘制合并后的图，仅显示整体R² ---
if all_true_values:
    plt.figure(figsize=(10, 7))
    plt.scatter(all_true_values, all_predicted_values, alpha=0.6)
    min_val = min(min(all_true_values), min(all_predicted_values))
    max_val = max(max(all_true_values), max(all_predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # 计算整体 R²
    overall_r2 = r2_score(all_true_values, all_predicted_values)

    # 标注整体 R²
    plt.text(0.05, 0.95, f'Overall R² = {overall_r2:.3f}',
             transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8))

    plt.title('True vs Predicted Values (All Columns Combined)')
    plt.xlabel('True Values (denormalized)')
    plt.ylabel('Predicted Values (denormalized)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{results_dir}All_Columns_Scatter_with_OverallR2.png', dpi=300)
    plt.close()

    # 保存做图数据为CSV（包含Column_Index）
    plot_data_df = pd.DataFrame({
        'Column_Index': all_col_labels,
        'True_Values': all_true_values,
        'Predicted_Values': all_predicted_values
    })
    plot_data_df.to_csv(f'{results_dir}做图数据.csv', index=False)
    print(f'Plotting data saved to: {results_dir}做图数据.csv')

print('\nProcessing completed!')
print(f'R² summary saved to: {results_dir}R2_Summary.csv')
