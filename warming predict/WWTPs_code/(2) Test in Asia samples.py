import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as func
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error



def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Linear_ANN(nn.Module):
    """
        Layer of our ANN.
    """

    def __init__(self, input_features, output_features, prior_var=1.):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize the weights and the bias
        self.weight = nn.Parameter(torch.randn(output_features, input_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_features))

    def forward(self, input):
        """
          Optimization process
        """
        return func.linear(input, self.weight, self.bias)


class Neural3network(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim, p=0):
        # call constructor from superclass
        super(Neural3network, self).__init__()

        # define network layers
        self.layer1 = Linear_ANN(in_dim, n_hidden_1)
        self.layer2 = Linear_ANN(n_hidden_1, out_dim)

        self.dropout = nn.Dropout(p)  # dropout训练

    def forward(self, x):
        # define forward pass
        x = x.view(x.size(0), -1)
        x = self.dropout(self.layer1(x))
        x = func.relu(x)
        x = torch.sigmoid(self.layer2(x))
        # x = func.relu(self.layer2(x))
        return x


class MyDataset(Dataset):
    def __init__(self, col=1):
        data1 = np.loadtxt('/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/污水厂数据集/Asia_metadata.csv', delimiter=',', skiprows=1,
                           usecols=range(1, 85), dtype=np.float32)
        data2 = np.loadtxt('/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/污水厂数据集/Asia_samples.csv', delimiter=',',
                           skiprows=1, usecols=col, dtype=np.float32)
        data2_normed = (data2 - data2.min(axis=0) + 1e-12) / (data2.max(axis=0) - data2.min(axis=0) + 1e-12)

        state = np.random.get_state()
        indices = np.arange(data1.shape[0])  
        np.random.shuffle(data1)
        np.random.set_state(state)
        np.random.shuffle(data2_normed)
        np.random.shuffle(indices)  

        self.features = torch.from_numpy(data1)
        self.targets = torch.reshape(torch.from_numpy(data2_normed), (182, 1))
        self.length = data1.shape[0]
        self.indices = torch.from_numpy(indices) 

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.indices[idx]  

def get_testdata(col=1):
    data = MyDataset(col)
    return data.features, data.targets, data.indices  


os.chdir('/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/Before Perturbed/alpha/')  # change direction.
os.getcwd()  # get current work direction.

para = np.ones((0, 91))
Test_result = np.ones((0, 91))

for col in range(1, 1626):
    for seed in range(0, 5):
        drop_P = 0
        wd = 0.01
        net = torch.load(
            str(col) + 'col-4fold-seed' + str(seed) + '-10000ep-78bS-0.00001lr-' + str(wd) + 'wd-drop' + str(
                drop_P) + '_train_network.pth', map_location='cpu')
        weight_H_To_O = np.diag(net.layer2.weight.data.numpy()[-1])
        weight_I_To_H = net.layer1.weight.data.numpy()
        weight_final = np.dot(weight_H_To_O, weight_I_To_H)
        abs_weight_final = abs(weight_final)
        weightForH = abs_weight_final / abs_weight_final.sum(axis=1, keepdims=True)  
        weightForH[np.isnan(weightForH)] = 0 
        Sum_input = weightForH.sum(axis=0)
        RI = Sum_input / Sum_input.sum() 
        para_bias = np.dot(net.layer2.weight.data.numpy(), net.layer1.bias.data.numpy()) + net.layer2.bias.data.numpy()
        seed_torch(seed)
        x_test, y_test, indices = get_testdata(col)  
        test_len = x_test.shape[0]
        col_result = np.ones((test_len, 1)) * col
        seed_result = np.ones((test_len, 1)) * seed
        wd_result = np.ones((test_len, 1)) * wd
        drop_P_result = np.ones((test_len, 1)) * drop_P
        net.eval()
        with torch.no_grad():
            Test_pred = net(x_test)
            Test_pred_ = Test_pred.data.numpy()
            y_test_ = y_test.numpy()
            x_test_ = x_test.numpy()
            MSE_T = mean_squared_error(y_test_, Test_pred_, sample_weight=None, multioutput='uniform_average')
            R2_T = r2_score(y_test_, Test_pred_)

        each_test = np.c_[indices, col_result, seed_result, wd_result, drop_P_result, x_test_, y_test_, Test_pred_]  
        Test_result = np.r_[Test_result, each_test]

        each = np.c_[col, seed, wd, drop_P, MSE_T, R2_T, RI.reshape(1, 84), para_bias]
        para = np.r_[para, each]

np.savetxt('/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/Before Perturbed/alpha/analysis/' + 'Test_Before.txt',
           Test_result, fmt='%.4e', delimiter='\t', newline='\n')
np.savetxt(
    '/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/Before Perturbed/alpha/analysis/' + 'Test_and_parameters_Before.csv',
    para, fmt='%.4e', delimiter=',', newline='\n')
