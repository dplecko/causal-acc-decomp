import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt

def cf_boost(train_data, eval_data, x_col, z_cols, w_cols, y_col, #lr=0.001,
             epochs=3, eta_de=0, eta_ie=0, eta_se_x1=0, eta_se_x0=0, constraints = ['Loss', 'NDE', 'NIE']):

    X_trn, Z_trn, W_trn, Y_trn = split_data(train_data, x_col, z_cols, w_cols, y_col, False)
    X_evl, Z_evl, W_evl, Y_evl = split_data(eval_data, x_col, z_cols, w_cols, y_col, False)

    fts_trn = torch.tensor(np.hstack([X_trn, Z_trn, W_trn]), dtype=torch.float)
    lbl_trn = torch.tensor(Y_trn, dtype=torch.float).unsqueeze(1)
    fts_evl = torch.tensor(np.hstack([X_evl, Z_evl, W_evl]), dtype=torch.float)
    lbl_evl = torch.tensor(Y_evl, dtype=torch.float).unsqueeze(1)

    task_type = 'regression' if len(np.unique(Y_trn)) > 2 else 'classification'

    px_z, px_wz = fit_propensity(X_trn, Z_trn, W_trn)
    px_z_evl, px_wz_evl = fit_propensity(X_evl, Z_evl, W_evl)

    F_t = nn_list()
    nde_hat = 0
    nie_hat = 0

    column_names = (
            [x_col] +
            (list(z_cols) if isinstance(z_cols, (list, np.ndarray)) else [z_cols]) +
            (list(w_cols) if isinstance(w_cols, (list, np.ndarray)) else [w_cols])
    )
    X_idx, Z_idx, W_idx = column_names.index(x_col), column_names.index(z_cols), column_names.index(w_cols)
    fts0_evl, fts1_evl = submodel_data(fts_evl, X_idx)

    nde_values = []
    nie_values = []

    for epoch in range(epochs):
        for constraint in constraints:

            if constraint in ["NDE", "NIE"]:
                fts_redt, fts_rede = fts_trn[:, Z_idx + W_idx], fts_evl[:, Z_idx + W_idx]
            elif constraint == "Loss":
                fts_redt, fts_rede = fts_trn[:, X_idx + Z_idx + W_idx], fts_evl[:, X_idx + Z_idx + W_idx]

            # initialize the network update
            h = TwoLayers(input_size=fts_redt.shape[1], hidden_size=16, output_size=1)

            grad = grad_L(F_t, fts_redt, px_z, px_wz, lbl_trn, constraint, nde_hat, eta_de, nie_hat, eta_ie)
            grad_evl = grad_L(F_t, fts_rede, px_z_evl, px_wz_evl, lbl_evl, constraint, nde_hat, eta_de, nie_hat, eta_ie)

            optimize_network(h, grad, grad_evl, fts_redt, fts_evl, X_idx, constraint=constraint)
            alpha_t = line_search(F_t, h, fts_redt, lbl_trn, X_idx, px_z, px_wz, eta_de,
                                  eta_ie, eta_se_x1, eta_se_x0, constraint, task_type, "Train")
            alpha_t_evl = line_search(F_t, h, fts_evl, lbl_evl, X_idx, px_z_evl, px_wz_evl, eta_de,
                                      eta_ie, eta_se_x1, eta_se_x0, constraint, task_type, "Eval")
            F_t.add_model(h, alpha_t)

            fitted = F_t.evaluate(fts_evl)
            fitted0 = F_t.evaluate(fts0_evl)
            fitted1 = F_t.evaluate(fts1_evl)
            nde_hat, nie_hat = causal_loss(fitted, fitted0, fitted1, fts_evl[:, X_idx], px_z_evl, px_wz_evl, eta_de,
                                           eta_ie, eta_se_x1, eta_se_x0, effect = "both", task_type = task_type)
            nde_values.append(nde_hat.item())
            nie_values.append(nie_hat.item())

            if constraint == "NDE":
                print("NDE achieved", nde_hat.item())
            elif constraint == "NIE":
                print("NIE achieved", nie_hat.item())

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(nde_values, label='NDE')
    plt.title('NDE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('NDE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(nie_values, label='NIE')
    plt.title('NIE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('NIE')
    plt.legend()

    plt.tight_layout()
    plt.show()
    return F_t

