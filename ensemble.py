#%%
import nbimporter
import numpy as np
import pandas as pd
#%%
from for_importing import X_cleaned
from binning_to_import import y
from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(X_cleaned[['racepctblack', 'racePctWhite', 'pctWInvInc', 'pctWPubAsst',
       'PctPopUnderPov', 'TotalPctDiv', 'PctKids2Par', 'PctPersOwnOccup']],y,test_size=0.2,random_state=22)
#%% md
# create models with all optimized parameters
#%%
from sklearn.linear_model import Lasso
x2tr=xtr[['racePctWhite','PctKids2Par']]
x2ts=xts[['racePctWhite','PctKids2Par']]
lin_reg=Lasso(alpha=2.6e-5,max_iter=1000,tol=.01)
lin_reg.fit(x2tr.values,ytr)

def linear_regression(x):
    if isinstance(x,pd.DataFrame):
        return lin_reg.predict(x[['racePctWhite', 'PctKids2Par']])
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return lin_reg.predict(x[:, [1, -2]])
#%%
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(C=1, solver='liblinear', class_weight={'0': 1, '1': 2})
log_reg.fit(xtr.values, ytr)

def logistic_regression(x):
    return log_reg.predict(x)
#%%
from sklearn import svm
svc = svm.SVC(C=4.72,gamma=.079,kernel='rbf',class_weight={'0':1,'1':4})
svc.fit(xtr.values, ytr)

def SVC(x):
    return svc.predict(x)

#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_dnn_model(train_loader, lr=0.001, epochs=100):
    model = DeepNN(input_dim=8)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            target = target.float().unsqueeze(1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return model


def DNN_predict(X, model):
    if isinstance(X, pd.Series):
        X = X.values.reshape(1, -1)
    elif isinstance(X, pd.DataFrame):
        X = X.values
    elif isinstance(X, np.ndarray) and X.ndim == 1:
        X = X.reshape(1, -1)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
        predictions = (output >= 0.5).float().squeeze().cpu().numpy().astype(int)

    # make output always a 1D numpy array
    if predictions.ndim == 0:
        return np.array([predictions])
    return predictions




ytr=ytr.astype(int)
yts=yts.astype(int)

from torch.utils.data import DataLoader, TensorDataset



Xtr_torch = torch.tensor(xtr.values, dtype=torch.float32)
Xts_torch = torch.tensor(xts.values, dtype=torch.float32)
ytr_torch = torch.tensor(ytr.values, dtype=torch.int32)
yts_torch = torch.tensor(yts.values, dtype=torch.int32)

train_dataset = TensorDataset(Xtr_torch, ytr_torch)
test_dataset = TensorDataset(Xts_torch, yts_torch)

batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_dataset = TensorDataset(Xtr_torch, ytr_torch)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# Train the model once
dnn_model = train_dnn_model(train_loader, lr=0.001, epochs=100)
#%%


def ensemble(x):
    if isinstance(x, pd.Series):
        x = x.values.reshape(1, -1)
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        x = x.reshape(1, -1)


    pred_logreg = logistic_regression(x).astype(int)
    pred_svc = SVC(x).astype(int)
    pred_dnn = DNN_predict(x, dnn_model)

    all_preds = np.vstack([pred_logreg, pred_svc, pred_dnn])

    pred_class = np.array([np.bincount(row).argmax() for row in all_preds.T])

    pred_regr = linear_regression(x)  # (n_samples,)
    
    return pred_class, pred_regr


def ensemble_accuracy(X, y):
    preds = []
    for i in range(len(X)):
        sample = X.iloc[i] if hasattr(X, 'iloc') else X[i]
        pred_class, _ = ensemble(sample)
        pred_scalar = pred_class[0] if isinstance(pred_class, (np.ndarray, list)) else pred_class
        preds.append(pred_scalar)

    y_array = y.astype(int).values if hasattr(y, 'astype') else y
    preds = np.array(preds)
    acc = (preds == y_array).mean() * 100
    return round(acc, 2)

#%%

#print(f'The ensemble has an accuracy of {ensemble_accuracy(xts, yts)}%')

#%%
