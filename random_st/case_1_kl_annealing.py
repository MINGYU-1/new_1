import os
import numpy as np
import pandas as pd
import sys
from vae_earlystopping import EarlyStopping
script_dir = os.path.dirname(os.path.abspath(__file__)) 
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
os.chdir(parent_dir)
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from model.m2_bce import BCEcVAE
from model.m2_mse import MSEcVAE
from loss.l2_bce import l2_bce
from loss.l2_mse import l2_mse
import joblib
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from kl_divergence.cyclical import gamma_cyclical
from kl_divergence.linear import gamma_linear
from kl_divergence.sigmoid import gamma_sigmoid
import json

results = {
    "R2_Sigmoid": [], 
    "R2_Linear": [],
    "R2_Cyclical":[]
}

for i in np.random.randint(1,100,size = 3):
    x_data = np.load('./data/metal.npy')
    c_data = np.load('./data/pre_re_fin.npy')
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

    train_data = [x_train,c_train]
    val_data = [x_val,c_val]
    test_data = [x_test,c_test]
    train_data = TensorDataset(*train_data)
    val_data = TensorDataset(*val_data)
    test_data = TensorDataset(*test_data)
    datas = [train_data,val_data,test_data]
    train_loader,val_loader,test_loader = [DataLoader(x,batch_size = 64,shuffle=False) for x in datas]
    x_sample,c_sample = next(iter(train_loader))

    x_dim = x_sample.shape[1]
    c_dim = c_sample.shape[1]
    x_dim,c_dim

    ### BCE모델에 대해서 정함
    model_bce = BCEcVAE(x_dim,c_dim,z_dim=8).to(device)
    early_stopping = EarlyStopping(patience=40,min_delta = 1e-9)
    optimizer = optim.Adam(model_bce.parameters(),lr = 1e-3, weight_decay=1e-5)
    epochs = 600
    ### train_val loader에서 Linear 학습
    gamma = gamma_linear(800, gamma_max=0.1, warmup_epochs=200)
    for epoch in range(1,epochs+1):
        model_bce.train()
        t_loss= 0
        for x,c in train_loader:
            x,c = x.to(device),c.to(device)
            optimizer.zero_grad()
            bce_logit, mu, logvar = model_bce(x,c)
            loss_dict = l2_bce(bce_logit, x,mu,logvar,beta=1,gamma = gamma)
            loss_dict['loss'].backward()
            optimizer.step()
            t_loss +=loss_dict['loss'].item()
        model_bce.eval()
        v_loss = 0
        with torch.no_grad():
            for v_x, v_c in val_loader:
                v_x,v_c = v_x.to(device),v_c.to(device)
                v_bce_logit, v_mu, v_logvar = model_bce(v_x,v_c)
                loss_dict = l2_bce(v_bce_logit, v_x, v_mu,v_logvar,beta=1,gamma = gamma)
                v_loss += loss_dict['loss'].item()
        avg_train_loss = t_loss/len(train_loader)
        avg_val_loss = v_loss/len(val_loader)

        if epoch % 20 ==0  or epoch ==2:
            print(f'Epoch [{epoch}/{epochs}]|Train:{avg_train_loss:.4f} |Val:{avg_val_loss:.4f}')
        if early_stopping(avg_val_loss,model_bce):
            break

    early_stopping.load_best_model(model_bce)
    model_bce.eval()

    ### MSE 구하는 방법
    x_sample,c_sample = next(iter(train_loader))
    x_dim = x_sample.shape[1]
    c_dim = c_sample.shape[1]
    x_dim,c_dim
    model_mse = MSEcVAE(x_dim,c_dim,z_dim=8).to(device)
    early_stopping = EarlyStopping(patience=40,min_delta = 1e-9)
    optimizer = optim.Adam(model_mse.parameters(),lr = 1e-3, weight_decay=1e-5)
    epochs = 800
    for epoch in range(1,epochs+1):
        model_mse.train()
        t_loss = 0
        for x,c in train_loader:
            x,c = x.to(device),c.to(device)
            x_hat,mu,logvar = model_mse(x,c)
            loss_dict = l2_mse(x_hat, x, mu, logvar, alpha = 1.0, gamma = gamma)
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()
            t_loss += loss_dict['loss'].item()
        v_loss = 0
        model_mse.eval()
        with torch.no_grad():
            for v_x,v_c in val_loader:
                v_x,v_c = v_x.to(device),v_c.to(device)
                x_hat,v_mu,v_logvar = model_mse(v_x,v_c)
                loss_dict = l2_mse(x_hat,v_x,v_mu,v_logvar, alpha = 1.0, gamma = gamma)
                v_loss += loss_dict['loss'].item()
        avg_train_loss = t_loss/len(train_loader)
        avg_val_loss = v_loss/len(val_loader)

        if epoch % 20 ==0  or epoch ==2:
            print(f'Epoch [{epoch}/{epochs}]|Train:{avg_train_loss:.4f} |Val:{avg_val_loss:.4f}')
        if early_stopping(avg_val_loss,model_mse):
            break
    
    early_stopping.load_best_model(model_mse)
    model_mse.eval()
    # evaluation
    all_bce_logits, all_mse_logits, all_true = [], [], []
    with torch.no_grad():
        for xt, ct in test_loader:
            xt, ct = xt.to(device), ct.to(device)
            b_logit, _, _ = model_bce(xt, ct)
            m_logit, _, _ = model_mse(xt, ct)
            all_bce_logits.append(b_logit.cpu().numpy())
            all_mse_logits.append(m_logit.cpu().numpy())
            all_true.append(xt.cpu().numpy())

    bce_logits = np.vstack(all_bce_logits)
    mse_logits = np.vstack(all_mse_logits)
    x_true_scaled = np.vstack(all_true)

    bce_prob = 1 / (1 + np.exp(-bce_logits))
    x_hat_fin = x_scaler.inverse_transform(mse_logits)
    x_true = x_scaler.inverse_transform(x_true_scaled)
    
    final_x_sig = x_hat_fin * bce_prob
    results["R2_Linear"].append(float(r2_score(x_true.flatten(), final_x_sig.flatten())))

    ##---Sigmoid에 대한 시도-----
    x_data = np.load('./data/metal.npy')
    c_data = np.load('./data/pre_re_fin.npy')
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

    train_data = [x_train,c_train]
    val_data = [x_val,c_val]
    test_data = [x_test,c_test]
    train_data = TensorDataset(*train_data)
    val_data = TensorDataset(*val_data)
    test_data = TensorDataset(*test_data)
    datas = [train_data,val_data,test_data]
    train_loader,val_loader,test_loader = [DataLoader(x,batch_size = 64,shuffle=False) for x in datas]
    x_sample,c_sample = next(iter(train_loader))

    x_dim = x_sample.shape[1]
    c_dim = c_sample.shape[1]
    x_dim,c_dim

    ### BCE모델에 대해서 정함
    model_bce = BCEcVAE(x_dim,c_dim,z_dim=8).to(device)
    early_stopping = EarlyStopping(patience=40,min_delta = 1e-9)
    optimizer = optim.Adam(model_bce.parameters(),lr = 1e-3, weight_decay=1e-5)
    epochs = 600
    ### train_val loader에서 Linear 학습
    gamma = gamma_sigmoid(800, gamma_max=0.1, center=300,steepness=0.3)
    for epoch in range(1,epochs+1):
        model_bce.train()
        t_loss= 0
        for x,c in train_loader:
            x,c = x.to(device),c.to(device)
            optimizer.zero_grad()
            bce_logit, mu, logvar = model_bce(x,c)
            loss_dict = l2_bce(bce_logit, x,mu,logvar,beta=1,gamma = gamma)
            loss_dict['loss'].backward()
            optimizer.step()
            t_loss +=loss_dict['loss'].item()
        model_bce.eval()
        v_loss = 0
        with torch.no_grad():
            for v_x, v_c in val_loader:
                v_x,v_c = v_x.to(device),v_c.to(device)
                v_bce_logit, v_mu, v_logvar = model_bce(v_x,v_c)
                loss_dict = l2_bce(v_bce_logit, v_x, v_mu,v_logvar,beta=1,gamma = gamma)
                v_loss += loss_dict['loss'].item()
        avg_train_loss = t_loss/len(train_loader)
        avg_val_loss = v_loss/len(val_loader)

        if epoch % 20 ==0  or epoch ==2:
            print(f'Epoch [{epoch}/{epochs}]|Train:{avg_train_loss:.4f} |Val:{avg_val_loss:.4f}')
        if early_stopping(avg_val_loss,model_bce):
            break

    early_stopping.load_best_model(model_bce)
    model_bce.eval()

    ### MSE 구하는 방법
    x_sample,c_sample = next(iter(train_loader))
    x_dim = x_sample.shape[1]
    c_dim = c_sample.shape[1]
    x_dim,c_dim
    model_mse = MSEcVAE(x_dim,c_dim,z_dim=8).to(device)
    early_stopping = EarlyStopping(patience=40,min_delta = 1e-9)
    optimizer = optim.Adam(model_mse.parameters(),lr = 1e-3, weight_decay=1e-5)
    epochs = 800
    for epoch in range(1,epochs+1):
        model_mse.train()
        t_loss = 0
        for x,c in train_loader:
            x,c = x.to(device),c.to(device)
            x_hat,mu,logvar = model_mse(x,c)
            loss_dict = l2_mse(x_hat, x, mu, logvar, alpha = 1.0, gamma = gamma)
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()
            t_loss += loss_dict['loss'].item()
        v_loss = 0
        model_mse.eval()
        with torch.no_grad():
            for v_x,v_c in val_loader:
                v_x,v_c = v_x.to(device),v_c.to(device)
                x_hat,v_mu,v_logvar = model_mse(v_x,v_c)
                loss_dict = l2_mse(x_hat,v_x,v_mu,v_logvar, alpha = 1.0, gamma = gamma)
                v_loss += loss_dict['loss'].item()
        avg_train_loss = t_loss/len(train_loader)
        avg_val_loss = v_loss/len(val_loader)

        if epoch % 20 ==0  or epoch ==2:
            print(f'Epoch [{epoch}/{epochs}]|Train:{avg_train_loss:.4f} |Val:{avg_val_loss:.4f}')
        if early_stopping(avg_val_loss,model_mse):
            break
    
    early_stopping.load_best_model(model_mse)
    model_mse.eval()
    # evaluation
    all_bce_logits, all_mse_logits, all_true = [], [], []
    with torch.no_grad():
        for xt, ct in test_loader:
            xt, ct = xt.to(device), ct.to(device)
            b_logit, _, _ = model_bce(xt, ct)
            m_logit, _, _ = model_mse(xt, ct)
            all_bce_logits.append(b_logit.cpu().numpy())
            all_mse_logits.append(m_logit.cpu().numpy())
            all_true.append(xt.cpu().numpy())

    bce_logits = np.vstack(all_bce_logits)
    mse_logits = np.vstack(all_mse_logits)
    x_true_scaled = np.vstack(all_true)

    bce_prob = 1 / (1 + np.exp(-bce_logits))
    x_hat_fin = x_scaler.inverse_transform(mse_logits)
    x_true = x_scaler.inverse_transform(x_true_scaled)
    
    final_x_sig = x_hat_fin * bce_prob
    results["R2_Sigmoid"].append(float(r2_score(x_true.flatten(), final_x_sig.flatten())))

        ##---Cyclical에 대한 시도-----
    x_data = np.load('./data/metal.npy')
    c_data = np.load('./data/pre_re_fin.npy')
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

    train_data = [x_train,c_train]
    val_data = [x_val,c_val]
    test_data = [x_test,c_test]
    train_data = TensorDataset(*train_data)
    val_data = TensorDataset(*val_data)
    test_data = TensorDataset(*test_data)
    datas = [train_data,val_data,test_data]
    train_loader,val_loader,test_loader = [DataLoader(x,batch_size = 64,shuffle=False) for x in datas]
    x_sample,c_sample = next(iter(train_loader))

    x_dim = x_sample.shape[1]
    c_dim = c_sample.shape[1]
    x_dim,c_dim

    ### BCE모델에 대해서 정함
    model_bce = BCEcVAE(x_dim,c_dim,z_dim=8).to(device)
    early_stopping = EarlyStopping(patience=40,min_delta = 1e-9)
    optimizer = optim.Adam(model_bce.parameters(),lr = 1e-3, weight_decay=1e-5)
    epochs = 600
    ### train_val loader에서 Linear 학습
    gamma = gamma_cyclical(800, gamma_max=0.1, cycle_len=200,ratio=0.5)
    for epoch in range(1,epochs+1):
        model_bce.train()
        t_loss= 0
        for x,c in train_loader:
            x,c = x.to(device),c.to(device)
            optimizer.zero_grad()
            bce_logit, mu, logvar = model_bce(x,c)
            loss_dict = l2_bce(bce_logit, x,mu,logvar,beta=1,gamma = gamma)
            loss_dict['loss'].backward()
            optimizer.step()
            t_loss +=loss_dict['loss'].item()
        model_bce.eval()
        v_loss = 0
        with torch.no_grad():
            for v_x, v_c in val_loader:
                v_x,v_c = v_x.to(device),v_c.to(device)
                v_bce_logit, v_mu, v_logvar = model_bce(v_x,v_c)
                loss_dict = l2_bce(v_bce_logit, v_x, v_mu,v_logvar,beta=1,gamma = gamma)
                v_loss += loss_dict['loss'].item()
        avg_train_loss = t_loss/len(train_loader)
        avg_val_loss = v_loss/len(val_loader)

        if epoch % 20 ==0  or epoch ==2:
            print(f'Epoch [{epoch}/{epochs}]|Train:{avg_train_loss:.4f} |Val:{avg_val_loss:.4f}')
        if early_stopping(avg_val_loss,model_bce):
            break

    early_stopping.load_best_model(model_bce)
    model_bce.eval()

    ### MSE 구하는 방법
    x_sample,c_sample = next(iter(train_loader))
    x_dim = x_sample.shape[1]
    c_dim = c_sample.shape[1]
    x_dim,c_dim
    model_mse = MSEcVAE(x_dim,c_dim,z_dim=8).to(device)
    early_stopping = EarlyStopping(patience=40,min_delta = 1e-9)
    optimizer = optim.Adam(model_mse.parameters(),lr = 1e-3, weight_decay=1e-5)
    epochs = 800
    for epoch in range(1,epochs+1):
        model_mse.train()
        t_loss = 0
        for x,c in train_loader:
            x,c = x.to(device),c.to(device)
            x_hat,mu,logvar = model_mse(x,c)
            loss_dict = l2_mse(x_hat, x, mu, logvar, alpha = 1.0, gamma = gamma)
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()
            t_loss += loss_dict['loss'].item()
        v_loss = 0
        model_mse.eval()
        with torch.no_grad():
            for v_x,v_c in val_loader:
                v_x,v_c = v_x.to(device),v_c.to(device)
                x_hat,v_mu,v_logvar = model_mse(v_x,v_c)
                loss_dict = l2_mse(x_hat,v_x,v_mu,v_logvar, alpha = 1.0, gamma = gamma)
                v_loss += loss_dict['loss'].item()
        avg_train_loss = t_loss/len(train_loader)
        avg_val_loss = v_loss/len(val_loader)

        if epoch % 20 ==0  or epoch ==2:
            print(f'Epoch [{epoch}/{epochs}]|Train:{avg_train_loss:.4f} |Val:{avg_val_loss:.4f}')
        if early_stopping(avg_val_loss,model_mse):
            break
    
    early_stopping.load_best_model(model_mse)
    model_mse.eval()
    # evaluation
    all_bce_logits, all_mse_logits, all_true = [], [], []
    with torch.no_grad():
        for xt, ct in test_loader:
            xt, ct = xt.to(device), ct.to(device)
            b_logit, _, _ = model_bce(xt, ct)
            m_logit, _, _ = model_mse(xt, ct)
            all_bce_logits.append(b_logit.cpu().numpy())
            all_mse_logits.append(m_logit.cpu().numpy())
            all_true.append(xt.cpu().numpy())

    bce_logits = np.vstack(all_bce_logits)
    mse_logits = np.vstack(all_mse_logits)
    x_true_scaled = np.vstack(all_true)

    bce_prob = 1 / (1 + np.exp(-bce_logits))
    x_hat_fin = x_scaler.inverse_transform(mse_logits)
    x_true = x_scaler.inverse_transform(x_true_scaled)
    
    final_x_sig = x_hat_fin * bce_prob
    results["R2_Cyclical"].append(float(r2_score(x_true.flatten(), final_x_sig.flatten())))
    
save_path = "./results_React_Presence.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)