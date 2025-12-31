import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import shap
import random
# 设置GPU
DEVICE = torch.device('cuda:0')  # GPU

def set_reproducible():  # 设置随机种子
    np.random.seed(0)  # 随机种子
    torch.manual_seed(0)  # 随机种子
    torch.backends.cudnn.deterministic = True  # 随机种子

set_reproducible()  # 设置随机种子

def calc_r_squared(y_true, y_pred): # 计算R2
    ss_residual = np.sum((y_true - y_pred) ** 2) # 残差平方和
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2) # 总平方和
    r_squared = 1 - (ss_residual / ss_total) # R2
    return r_squared # 返回R2

def calc_mare(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def calc_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# df_raw = pd.read_csv('F:\\yuancian\\2017182_test.csv')
df_raw = pd.read_csv('6_4_3p_submonthtime_sorted.csv')

num_train = df_raw[(df_raw.Year>=2003)&(df_raw.Year<2013)].index 
num_val = df_raw[(df_raw.Year>=2013)&(df_raw.Year<2016)].index   
num_test = df_raw[(df_raw.Year>=2016)&(df_raw.Year<=2018)].index
# print("num_test的个数为:", len(num_test))

col_names = []
for name in ['Lon','Lat','DEM','MODD','MODN','MYDD','MYDN','Srad','Diwu','EVI2_','GPM','SM','DOY']:
    col_names += [name + str(i) for i in range(9, 16)]
x_train = df_raw.loc[num_train, col_names].values  # 训练集特征

# 训练集目标变量
y_train = df_raw.loc[num_train, 'TA'].values

from sklearn.preprocessing import StandardScaler
TA_scaler = StandardScaler()
y_train = TA_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

x_val = df_raw.loc[num_val, col_names].values  
y_val = TA_scaler.fit_transform(df_raw.loc[num_val, 'TA'].values.reshape(-1, 1)).flatten()  

scaler = StandardScaler()  # 标准化

xt_train = torch.from_numpy(x_train[:300].reshape(300, 7, 13)).float().to(DEVICE)  
yt_train = torch.from_numpy(y_train[:300]).float().to(DEVICE)  
# print(f"[xt_train] x.shape: {xt_train.shape}") 

# 修改prepare_tensor函数
def prepare_tensor(x, y):  # 转换为张量
    return (
        torch.from_numpy(x.reshape(-1, 7, 13)).float().to(DEVICE),  # 转换为张量
        torch.from_numpy(y).float().to(DEVICE)  # 转换为张量
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerCNN(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=1, num_layers=1, hidden_size=512, l2_lambda=0.01):
        super(TransformerCNN, self).__init__()

        # Transformer Encoder
        self.input_layer = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # CNN Layer
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()

        # Output Layer
        self.output_layer = nn.Linear(hidden_size, output_dim)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        # Transformer Encoder
        # print(f"[输入] x.shape: {x.shape}") 
        x = self.input_layer(x)
        # print(f"[线性映射后] x.shape: {x.shape}")
        x = self.pos_encoder(x)
        # print(f"[位置编码后] x.shape: {x.shape}") 
        # x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        transformer_out = self.transformer_encoder(x)
        # print(f"[Transformer输出] transformer_out.shape: {transformer_out.shape}")
        # CNN Forward Pass
        transformer_out = transformer_out.permute(0, 2, 1)  # (batch_size, d_model, seq_len)
        # print(f"[CNN输入] x_cnn.shape: {transformer_out.shape}")
        conv_out = self.conv1(transformer_out)
        
        conv_out = self.bn1(conv_out)
        conv_out = self.relu(conv_out)
        # print(f"[CNN后的] conv_out.shape: {conv_out.shape}") 
        # Global average pooling
        pooled = conv_out.mean(dim=2)  # (batch_size, hidden_size)
        # print(f"[池化后的] pooled.shape: {pooled.shape}") 
        # Final output layer
        output = self.output_layer(pooled)
        # print(f"[输出] output.shape: {output.shape}") 
        # L2 Regularization
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        l2_loss = self.l2_lambda * l2_loss

        return output.squeeze(-1), l2_loss



# 设置参数
input_dim = xt_train.shape[2]  

output_dim = 1  # 目标数
d_model = 512
nhead = 8
num_layers = 1
l2_lambda = 0.01

net = TransformerCNN(input_dim, output_dim, d_model, nhead, num_layers) # 实例化模型
net.to(DEVICE) # 模型放入GPU

BSZ = 4096*4*2 # 设置批量大小
lr = 0.0001 # 设置学习率
momentum = 0.9 # 设置动量

criterion = nn.MSELoss() 
optimizer = optim.Adam(net.parameters(), lr=lr) 


def calc_test_pred(net, x, y, batch_size=512):
    net.eval()
    yt_pred_batch_list = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xt_batch, yt_batch = prepare_tensor(x[i:i + batch_size], y[i:i + batch_size])
            yt_pred_batch, _ = net(xt_batch)
            yt_pred_batch_list.append(yt_pred_batch)
    y_pred = torch.cat(yt_pred_batch_list, dim=0).cpu().numpy()
    return y_pred

def calc_pred(net, x, y):
    net.eval() 
    yt_pred_batch_list = [] 
    with torch.no_grad(): 
        for i in range(0, x.shape[0], BSZ): 
            xt, yt = prepare_tensor(x[i:i + BSZ], y[i:i + BSZ]) 
            yt_pred_batch, _ = net(xt)  
            yt_pred_batch_list.append(yt_pred_batch) 
    y_pred = torch.cat(yt_pred_batch_list, dim=0).cpu().numpy() 
    net.train() 
    return y_pred 

def calc_all_indicators(y_true, y_pred): 
    mse = mean_squared_error(y_true, y_pred) 
    rmse = np.sqrt(mse) 
    return mse, rmse 

loss_train_list, rmse_train_list, rmse_val_list, r2_val_list,r2_train_list,mae_train_list, mae_val_list= [], [], [], [],[],[],[] 

for epoch in range(1,169): 
    net.train() 
    for i in range(0, x_train.shape[0], BSZ): 
        xt_train, yt_train = prepare_tensor(x_train[i:i + BSZ], y_train[i:i + BSZ]) #
        optimizer.zero_grad() 
        outputs, l2_loss = net(xt_train) 
        loss_batch = criterion(outputs, yt_train) + l2_loss 
        loss_batch.backward() 
        optimizer.step() 

    y_train_pred = calc_pred(net, x_train, y_train) 
    y_train_inverse_pre = TA_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()  
    y_train_inverse = TA_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

    loss_train ,_ = calc_all_indicators(y_train, y_train_pred)
    _, rmse_train = calc_all_indicators(y_train_inverse, y_train_inverse_pre) 
    r2_train = calc_r_squared(y_train_inverse, y_train_inverse_pre) 
    r2_train_list.append(r2_train) 
    loss_train_list.append(loss_train)
    rmse_train_list.append(rmse_train)
    mae_train = calc_mae(y_train_inverse, y_train_inverse_pre)
    mae_train_list.append(mae_train)
    y_val_pred = calc_pred(net, x_val, y_val) 
    y_val_inverse_pre = TA_scaler.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()  
    y_val_inverse = TA_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    mae_val = calc_mae(y_val_inverse, y_val_inverse_pre)
    mae_val_list.append(mae_val)
    _, rmse_val = calc_all_indicators(y_val_inverse, y_val_inverse_pre)
    rmse_val_list.append(rmse_val) 
    r2_val = calc_r_squared(y_val_inverse, y_val_inverse_pre) 
    r2_val_list.append(r2_val) 
    print('[epoch {:d}] train MAE: {:.3f}, val MAE: {:.3f}, training rmse: {:.3f}, val rmse: {:.3f}, train R^2: {:.3f}, val R^2: {:.3f}'.format(epoch, mae_train, mae_val, rmse_train, rmse_val, r2_train, r2_val)) # 修改此处，添加训练集 R^2

    if (r2_val>0.2): 
        torch.save(net.state_dict(), 'hb_ah_js_7/1_TC.pth') 


        os.makedirs('hb_ah_js_7', exist_ok=True)
        joblib.dump(TA_scaler, 'hb_ah_js_7/1_TA_scaler.pkl')

model=TransformerCNN(input_dim, output_dim, d_model, nhead, num_layers)


model.load_state_dict(torch.load('hb_ah_js_7/1_TC.pth'))
model.eval()

x_test = df_raw.loc[num_test, col_names].values  

import joblib, os
TA_scaler = joblib.load('hb_ah_js_7/1_TA_scaler.pkl') 
y_test = TA_scaler.transform(df_raw.loc[num_test, 'TA'].values.reshape(-1, 1)).flatten()  

xt_test = torch.from_numpy(x_test.reshape(-1, 7, 13)).float().to(DEVICE)  


yt_test = torch.from_numpy(y_test).float().to(DEVICE)  

y_test_pred = calc_test_pred(net, x_test, y_test,batch_size=1024)

wanted_pred_t = TA_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

y_test_inverse_pre = TA_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten() 
y_test_inverse = TA_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

mse_test, rmse_test = calc_all_indicators(y_test_inverse, y_test_inverse_pre)  
r2_test = calc_r_squared(y_test_inverse, y_test_inverse_pre)  
r_test = np.sqrt(r2_test) 
mae_test = calc_mae(y_test_inverse, y_test_inverse_pre)


print('测试集结果:')
print('Test MSE: {:.3f}'.format(mse_test))
print('Test RMSE: {:.3f}'.format(rmse_test))
print('Test MAE: {:.3f}'.format(mae_test))
print('Test R^2: {:.3f}'.format(r2_test))
print('Test R: {:.3f}'.format(r_test)) 

pd.DataFrame({'Observed': y_test_inverse, 'Predicted': y_test_inverse_pre}).to_csv('hb_ah_js_7/2016-2019TC-TA.csv', header=None, index=None)

