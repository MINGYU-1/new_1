import os
import numpy as np
import pandas as pd
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) 
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from vae_earlystopping import EarlyStopping
from model_1.m1 import Model1
from loss_1.l1 import loss1
import joblib
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
### 구하기
import json

results = {
    "random_state": [],
    "R2_BINARY": [],      # bce_binary * mse (최종)
    "R2_BCE_MSE": [],     # bce_prob * mse (sigmoid 가중)
    "R2_MSE": []          # mse만 (x_hat_fin)
}
for i in np.random.randint(1,100,size = 20):
    x_data = np.load('./data/metal.npy')
    c_data = np.load('./data/pre_re.npy')
    x_train,x_test,c_train,c_test = train_test_split(x_data,c_data, random_state = i,test_size = 0.4)
    x_val,x_test,c_val,c_test = train_test_split(x_test,c_test,random_state = i, test_size = 0.5)
    x_scaler = MinMaxScaler()
    c_scaler = MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train)
    c_train = c_scaler.fit_transform(c_train)
    x_val,x_test = [x_scaler.transform(x) for x in [x_val,x_test]]
    c_val,c_test = [c_scaler.transform(c) for c in [c_val,c_test]]

    x_train,x_val,x_test = [torch.tensor(x, dtype = torch.float32) for x in [x_train,x_val,x_test]]
    c_train,c_val,c_test = [torch.tensor(c, dtype = torch.float32) for c in [c_train,c_val,c_test]]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = [x_train,x_train,c_train,c_train]
    val_data = [x_val,x_val,c_val,c_val]
    test_data = [x_test,x_test,c_test,c_test]
    train_data = TensorDataset(*train_data)
    val_data = TensorDataset(*val_data)
    test_data = TensorDataset(*test_data)
    datas = [train_data,val_data,test_data]
    train_loader,val_loader,test_loader = [DataLoader(x,batch_size = 64,shuffle=False) for x in datas]
    x_sample,x2_sample, c_sample,c2_sample = next(iter(train_loader))

    x_dim = x_sample.shape[1]
    x2_dim= x2_sample.shape[1]
    c_dim = c_sample.shape[1]
    c2_dim = c2_sample.shape[1]

    model = Model1(x_dim,x2_dim,c_dim,c2_dim, z_dim=8).to(device)
    early_stopping = EarlyStopping(patience=40,min_delta = 1e-9)
    optimizer = optim.Adam(model.parameters(),lr = 1e-3, weight_decay=1e-5)
    epochs = 600
    ### train_val loader에서의 학습
    for epoch in range(1,epochs+1):
        model.train()
        t_loss= 0
        for x,x2, c, c2 in train_loader:
            x,x2,c,c2 = x.to(device),x2.to(device),c.to(device),c2.to(device)
            optimizer.zero_grad()
            bce_logit, mu, logvar = model(x,x2,c,c2)
            loss_dict = loss1(bce_logit,x_hat,x,x2,mu,logvar)
            loss_dict['loss'].backward()
            optimizer.step()
            t_loss +=loss_dict['loss'].item()
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for v_x,v2_x, v_c,v2_c in val_loader:
                v_x,v2_x,v_c,v2_c = v_x.to(device),v2_x.to(device),v_c.to(device),v2_c.to(device)
                v_bce_logit,v_x_hat, v_mu, v_logvar = model(v_x,v2_x,v_c,v2_c)
                loss_dict = loss1(v_bce_logit,v_x_hat, v_x,v2_x, v_mu,v_logvar)
                v_loss += loss_dict['loss'].item()
        avg_train_loss = t_loss/len(train_loader)
        avg_val_loss = v_loss/len(val_loader)

        if epoch % 20 ==0  or epoch ==2:
            print(f'Epoch [{epoch}/{epochs}]|Train:{avg_train_loss:.4f} |Val:{avg_val_loss:.4f}')
        if early_stopping(avg_val_loss,model):
            break


### mse
    
    early_stopping.load_best_model(model)
    model.eval()
    mse_logit_list = []
    x_true_list = []
    with torch.no_grad():
        for x_t, x2_t,c_t,c2_t in test_loader:
            x_t,x2_t,c_t,c2_t = x_t.to(device),x2_t.to(device), c_t.to(device)c2_t.to(device)
            bce_logit, x_hat,mu_t,logvar_t = model(x_t,x2_t,c_t,c2_t)
            mse_logit_list.append(x_hat.cpu().numpy())
            x_true_list.append(x_t.cpu().numpy())
    mse_logits = np.vstack(mse_logit_list)
    x_true = np.vstack(x_true_list)
    ### BCE-> 0,1 표현하여 MSE에 곱하기
    all_x_hat_tensor = torch.cat(all_x_hat, dim=0) # (Total_Samples, x_dim) 형태로 결합
    bce_logits_np = all_x_hat_tensor.cpu().numpy()
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    bce_prob = sigmoid(bce_logits_np)
    bce_binary = (bce_prob >= 0.5).astype(np.float32)
    x_hat_fin = x_scaler.inverse_transform(mse_logits)
    x_true = x_scaler.inverse_transform(x_true)
    final_x_hat = x_hat_fin*bce_binary
    final_x_sig = x_hat_fin*bce_prob
    from sklearn.metrics import r2_score,mean_squared_error
    r2_mse = r2_score(x_true.flatten(),x_hat_fin.flatten())
    r2_bce_mse = r2_score(x_true.flatten(), final_x_hat.flatten())
    r2_bce_mse_sig = r2_score(x_true.flatten(),final_x_sig.flatten())
    ##각각의 r2저장
    results["random_state"].append(int(i))
    results["R2_BINARY"].append(float(r2_bce_mse))
    results["R2_BCE_MSE"].append(float(r2_bce_mse_sig))
    results["R2_MSE"].append(float(r2_mse))

save_path = "./results_r2.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Saved:", save_path)