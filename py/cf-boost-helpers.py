import pdb
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.nn.functional as F
import copy

class TwoLayers(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayers, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

class nn_list:
    def __init__(self):
        self.models = []
        self.alphas = []

    def add_model(self, model, alpha):
        self.models.append(model)
        self.alphas.append(alpha)

    def evaluate(self, x):
        # Assumes x is a PyTorch tensor
        output = 0
        for model, alpha in zip(self.models, self.alphas):
            output += alpha * model(x)
        return output

    def predict(self, df):
        # Converts DataFrame to PyTorch tensor and evaluates
        x_tensor = torch.from_numpy(df.values).float()  # Ensure conversion to tensor with appropriate type
        return self.evaluate(x_tensor)

def fit_propensity(X, Z, W):

    zlog_reg = LogisticRegression()
    zlog_reg.fit(Z, X.ravel())
    wzlog_reg = LogisticRegression()
    wzlog_reg.fit(np.hstack([Z, W]), X.ravel())

    px_z = torch.tensor(zlog_reg.predict_proba(Z)[:, 1], dtype=torch.float)
    px_wz = torch.tensor(wzlog_reg.predict_proba(np.hstack([Z, W]))[:, 1], dtype=torch.float)

    return px_z, px_wz

def split_data(data, x_col, z_cols, w_cols, y_col, tensor=True):
    if tensor:
        X = torch.tensor(data[x_col].values.reshape(-1, 1), dtype=torch.float)
        Z = torch.tensor(data[z_cols].values.reshape(-1, len(z_cols)), dtype=torch.float)
        W = torch.tensor(data[w_cols].values.reshape(-1, len(w_cols)), dtype=torch.float)
        Y = torch.tensor(data[y_col].values.reshape(-1, 1).squeeze(), dtype=torch.float)
    else:
        X = data[x_col].values.reshape(-1, 1)
        Z = data[z_cols].values.reshape(-1, len(z_cols))
        W = data[w_cols].values.reshape(-1, len(w_cols))
        Y = data[y_col].values.reshape(-1, 1).squeeze()
    return X, Z, W, Y

def submodel_data(fts, x_idx):
    fts0 = fts.clone()
    fts1 = fts.clone()
    fts0[:, x_idx] = 0
    fts1[:, x_idx] = 1

    return fts0, fts1

def grad_L(F_t, fts, px_z, px_wz, lbl, constraint, nde_hat, eta_1, nie_hat, eta_2):

    if constraint == "Loss":
        wgh = 2 / len(lbl) * (lbl - F_t.evaluate(fts))
    elif constraint == "NDE":
        scl_fct = -2 / len(lbl) * (nde_hat - eta_1)
        wgh = scl_fct * (1 - px_wz) / (1 - px_z)  # Placeholder for NDE computation
    elif constraint == "NIE":
        scl_fct = -2 / len(lbl) * (nie_hat - eta_2)
        wgh = scl_fct * (px_wz / px_z - (1 - px_wz) / (1 - px_z))  # Placeholder for NIE computation
    else:
        wgh = np.zeros_like(lbl)  # Default case if needed

    if hasattr(wgh, 'detach'):
        wgh = wgh.detach()

    return wgh

def causal_loss(pred, pred0, pred1, X, px_z, px_wz, eta_de, eta_ie, eta_se_x1,
                eta_se_x0, effect = "NDE", task_type = "regression"):
    if task_type == 'classification':
        pred_prob = torch.sigmoid(pred).squeeze()
        pred_prob1 = torch.sigmoid(pred1).squeeze()
        pred_prob0 = torch.sigmoid(pred0).squeeze()
    else:
        pred_prob = pred.squeeze()
        pred_prob1 = pred1.squeeze()
        pred_prob0 = pred0.squeeze()
    X_sq = X.squeeze()

    # get P(x | z) model
    px = X.mean()

    # get f_{x_1, W_{x_0}}
    wgh0 = (1 - px_wz) / (1 - px_z)
    fx1_wx0 = (pred_prob1 * wgh0).sum() / (wgh0.sum())

    # get f_{x_0, W_{x_0}}
    fx0_wx0 = (pred_prob0 * wgh0).sum() / (wgh0.sum())

    # get f_{x_1, W_{x_1}}
    wgh1 = px_wz / (px_z)
    fx1_wx1 = (pred_prob1 * wgh1).sum() / (wgh1.sum())

    # get f | x0
    f_x0 = pred_prob[X_sq == 0].mean()

    # get f | x1
    f_x1 = pred_prob[X_sq == 1].mean()

    # \sum_i=1^n [f(x1, w) - f(x0, w)] * 1 / n (direct effect)
    nde_loss = torch.abs(fx1_wx0 - fx0_wx0 - eta_de) ** 2
    nie_loss = torch.abs(fx1_wx0 - fx1_wx1 - eta_ie) ** 2
    # nse_loss_x1 = torch.abs(f_x1 - fx1_wx1 - eta_se_x1)
    # nse_loss_x0 = torch.abs(f_x0 - fx0_wx0 - eta_se_x0)

    if effect == "NDE":
        return nde_loss
    elif effect == "NIE":
        return nie_loss
    elif effect == "both":
        return fx1_wx0 - fx0_wx0, fx1_wx0 - fx1_wx1


def ternary_search(func, left, right, eps=1e-5):
    """
    Perform a ternary search between left and right to minimize the function `func`.
    `eps` is the precision of the search.
    """
    while right - left > eps:
        l_third = left + (right - left) / 3
        r_third = right - (right - left) / 3
        if func(l_third) < func(r_third):
            right = r_third
        else:
            left = l_third
    return (left + right) / 2


def line_search(F_t, h_t, fts, lbl, x_idx, px_z, px_wz, eta_de, eta_ie, eta_se_x1, eta_se_x0, constraint, task_type,
                fold="train"):

    X = fts[:, x_idx]
    fts0, fts1 = submodel_data(fts, x_idx)

    fitted, fitted0, fitted1 = F_t.evaluate(fts), F_t.evaluate(fts0), F_t.evaluate(fts1)
    delta, delta0, delta1 = h_t(fts), h_t(fts0), h_t(fts1)
    alpha_min_loss = float('inf')
    best_alpha = 0  # Initialize best_alpha

    alpha_vals = []
    loss_vals = []

    for i in range(-5, 3):  # 10^-5 to 10^2
        alpha_t = 10 ** i

        if constraint == "Loss":
            alpha_t_val = nn.MSELoss()(fitted + alpha_t * delta, lbl)
        else:
            alpha_t_val = causal_loss(fitted + alpha_t * delta, fitted0 + alpha_t * delta0, fitted1 + alpha_t * delta1,
                                      X, px_z, px_wz, eta_de, eta_ie, eta_se_x1, eta_se_x0,
                                      effect=constraint, task_type=task_type)

        alpha_vals.append(alpha_t)
        loss_vals.append(alpha_t_val.item())

        if alpha_t_val < alpha_min_loss:
            alpha_min_loss = alpha_t_val
            best_alpha = alpha_t

    if constraint == "NDE":
        pdb.set_trace()

    if False:
        plt.figure(figsize=(8, 4))
        plt.plot(log(alpha_vals), loss_vals, marker='o')
        plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.xlabel('Alpha')
        plt.ylabel('Loss')
        plt.title(f'{constraint} Loss vs. Alpha')
        plt.grid(True)
        plt.show()

    # Ternary search for more precise alpha between neighbors
    best_i = np.log10(best_alpha)
    left, right = 10 ** (best_i - 1), 10 ** (best_i + 1)

    def func(alpha_t):
        if constraint == "Loss":
            return nn.MSELoss()(fitted + alpha_t * delta, lbl).item()
        else:
            #' * Adjust this to be a function of alpha_t (!!!) *
            return causal_loss(fitted, fitted0, fitted1, X, px_z, px_wz, eta_de, eta_ie, eta_se_x1, eta_se_x0,
                               effect=constraint, task_type=task_type).item()

    best_alpha = ternary_search(func, left, right)

    print(fold, constraint, "optim: Best alpha", best_alpha, "with loss", alpha_min_loss.item())
    return best_alpha

def optimize_network(h, grad, grad_evl, fts, fts_evl, x_idx,
                     lr=0.01, epochs=500, constraint="Loss", batch_size=256, patience=10):

    optimizer = torch.optim.Adam(h.parameters(), lr=lr)
    best_loss = float('inf')
    epochs_no_improve = 0

    fts0_evl, fts1_evl = submodel_data(fts_evl, x_idx)

    # TensorDataset and DataLoader for training
    dataset = torch.utils.data.TensorDataset(fts, grad)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_model_state = copy.deepcopy(h.state_dict())

    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()

            if constraint == "Loss":
                outputs = h(inputs)
            elif constraint == "NDE":
                inputs0, inputs1 = submodel_data(inputs, x_idx)
                outputs = h(inputs1) - h(inputs0) # h(inputs)
            elif constraint == "NIE":
                _, inputs1 = submodel_data(inputs, x_idx)
                outputs = h(inputs1) # h(inputs)

            loss = -torch.dot(targets.squeeze(), outputs.squeeze()) / torch.norm(outputs, p=2)
            loss.backward()
            optimizer.step()

        # Early stopping based on evaluation data
        with torch.no_grad():
            if constraint == "Loss":
                evl_outputs = h(fts_evl)
            elif constraint == "NDE":
                evl_outputs = h(fts1_evl) - h(fts0_evl)
            elif constraint == "NIE":
                evl_outputs = h(fts1_evl)

            evl_loss = -torch.dot(grad_evl.squeeze(), evl_outputs.squeeze()) / torch.norm(evl_outputs, p=2)
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch}: Eval Loss = {evl_loss.item()}")
            if evl_loss < best_loss:
                best_loss = evl_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(h.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    h.load_state_dict(best_model_state)
                    break
