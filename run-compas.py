from sklearn.model_selection import train_test_split
import os
import pandas as pd
cur_dir = os.getcwd()
rel_path = os.path.join(cur_dir, 'py')
for file in os.listdir(rel_path):
    if file.endswith(".py"):
        exec(open(os.path.join(rel_path, file)).read())

if __name__ == "__main__":
    compas_data = pd.read_csv("~/Desktop/compas_regression.csv")
    compas_trn, compas_evl = train_test_split(compas_data, test_size=0.3, random_state=42)
    epochs = 3
    constraints = ['Loss', 'NDE', 'NIE']
    x_col = ['race']
    y_col = ['two_year_recid']
    w_cols = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree']
    z_cols = ['sex', 'age']
    booster = cf_boost(compas_trn, compas_evl,
                       x_col = x_col, y_col = y_col, w_cols = w_cols, z_cols= z_cols,
                       epochs = epochs, constraints = constraints)

    booster.predict(compas_evl[x_col + z_cols + w_cols])