import os, sys, torch, joblib, json
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# 1. Setup paths BEFORE importing custom modules
script_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# 사용자 정의 모듈 임포트
from vae_earlystopping import EarlyStopping
from model1.m2_bce import BCEcVAE
from model1.m2_mse import MSEcVAE
from loss1.l2_bce import l2_bce
from loss1.l2_mse import l2_mse
from kl_divergence.cyclical import gamma_cyclical
from kl_divergence.linear import gamma_linear
from kl_divergence.sigmoid import gamma_sigmoid

def train_and_eval(gamma_obj, train_loader, val_loader, test_loader, x_scaler, x_dim, c_dim, device):
    # 1. BCE 모델 학습
    model_bce = BCEcVAE(x_dim, c_dim, z_dim=8).to(device)
    optimizer = optim.Adam(model_bce.parameters(), lr=1e-3, weight_decay=1e-5)
    early_stopping = EarlyStopping(patience=40, min_delta=1e-9)
    
    for epoch in range(1, 801):
        model_bce.train()
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model_bce(x, c)
            loss = l2_bce(logits, x, mu, logvar, beta=1, gamma=gamma_obj)['loss']
            loss.backward(); optimizer.step()
        
        # Validation (Early Stopping) 생략 가능하지만 유지 시 여기에 추가
        if early_stopping(loss.item(), model_bce): break # 단순화를 위해 학습 loss 활용 예시

    # 2. MSE 모델 학습 (동일 구조)
    model_mse = MSEcVAE(x_dim, c_dim, z_dim=8).to(device)
    optimizer = optim.Adam(model_mse.parameters(), lr=1e-3, weight_decay=1e-5)
    early_stopping = EarlyStopping(patience=40, min_delta=1e-9)

    for epoch in range(1, 801):
        model_mse.train()
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model_mse(x, c)
            loss = l2_mse(x_hat, x, mu, logvar, alpha=1.0, gamma=gamma_obj)['loss']
            loss.backward(); optimizer.step()
        if early_stopping(loss.item(), model_mse): break

    # 3. 평가 (R2 Score)
    model_bce.eval(); model_mse.eval()
    all_bce, all_mse, all_true = [], [], []
    with torch.no_grad():
        for xt, ct in test_loader:
            xt, ct = xt.to(device), ct.to(device)
            b_logit, _, _ = model_bce(xt, ct)
            m_logit, _, _ = model_mse(xt, ct)
            all_bce.append(b_logit.cpu().numpy())
            all_mse.append(m_logit.cpu().numpy())
            all_true.append(xt.cpu().numpy())

    bce_prob = 1 / (1 + np.exp(-np.vstack(all_bce)))
    mse_res = x_scaler.inverse_transform(np.vstack(all_mse))
    x_true = x_scaler.inverse_transform(np.vstack(all_true))
    
    final_pred = mse_res * bce_prob
    return float(r2_score(x_true.flatten(), final_pred.flatten()))

# --- 메인 실행 루프 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = {"R2_Constant":[],"R2_Sigmoid": [], "R2_Linear": [], "R2_Cyclical": []}

data_dir = os.path.join(parent_dir, 'data') # parent_dir 내의 data 폴더 지정

# 파일 로드 부분 수정
x_data = np.load(os.path.join(data_dir, 'metal.npy'))
c_data = np.load(os.path.join(data_dir, 'pre_re_fin.npy'))

for seed in np.random.randint(1, 100, size=20):
    # 데이터 분할 및 스케일링 (중복 제거)
    x_train, x_test, c_train, c_test = train_test_split(x_data, c_data, random_state=seed, test_size=0.4)
    x_val, x_test, c_val, c_test = train_test_split(x_test, c_test, random_state=seed, test_size=0.5)
    
    xs, cs = MinMaxScaler(), MinMaxScaler()
    x_train_s = xs.fit_transform(x_train); c_train_s = cs.fit_transform(c_train)
    x_val_s, x_test_s = xs.transform(x_val), xs.transform(x_test)
    c_val_s, c_test_s = cs.transform(c_val), cs.transform(c_test)

    # Loader 생성 함수화하여 사용 (생략 가능)
    train_l = DataLoader(TensorDataset(torch.tensor(x_train_s, dtype=torch.float32), torch.tensor(c_train_s, dtype=torch.float32)), batch_size=64)
    val_l = DataLoader(TensorDataset(torch.tensor(x_val_s, dtype=torch.float32), torch.tensor(c_val_s, dtype=torch.float32)), batch_size=64)
    test_l = DataLoader(TensorDataset(torch.tensor(x_test_s, dtype=torch.float32), torch.tensor(c_test_s, dtype=torch.float32)), batch_size=64)

    x_dim, c_dim = x_train.shape[1], c_train.shape[1]

    # 각 스케줄러 실행
    schedules = {
        "R2_Constant":0.1,
        "R2_Linear": gamma_linear(800, 0.1, 200),
        "R2_Sigmoid": gamma_sigmoid(800, 0.1, 300, 0.3),
        "R2_Cyclical": gamma_cyclical(800, 0.1, 120, 0.5)
    }

    for key, gamma_obj in schedules.items():
        score = train_and_eval(gamma_obj, train_l, val_l, test_l, xs, x_dim, c_dim, device)
        results[key].append(score)

# 결과 저장
with open("./results_Kl_Annealing.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)