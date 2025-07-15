"""
This script loads and preprocesses a dataset, trains a machine learning model, and evaluates its performance.
Dependencies and configurations are centralized in `config.py`.
"""

import os, sys
sys.path.append(os.getcwd())
from config import *  # Import everything from config.py
from functions.BTN_KM import btnkm

df = pd.read_csv("data/spambase.csv", header=None)
df.columns = df.iloc[0]  
df = df[1:]  
df.reset_index(drop=True, inplace=True)  
df = df.values
df = df.astype(float)

X = df[:, :57] #features 
y = df[:, 57] #target


y = np.array(
    [float(y[0]) if isinstance(y, (list, np.ndarray)) else float(y) for y in y]
)
y = np.where(y > 0, 1, -1) 

# hyper-parameters
input_dimension = 30
max_rank = 25

a, b = 1e-2, 1e-3
c, d = 1e-5*np.ones(max_rank), 1e-6*np.ones(max_rank)
g, h =  1e-6*np.ones(input_dimension), 1e-6*np.ones(input_dimension)

# Store the MSE results
mse_values = []
R_effective = []
nll_values = []

# Loop 10 times to run the training and evaluation
start_time = time.time() 
for i in range(10):
    # Split the data
    np.random.seed(i)
    indices = np.random.permutation(len(X)) 
    split_index = int(0.90 * len(X))  # 90% for training, 10% for testing
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]
    # === Normalize X_train column-wise ===
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std  # Use train stats

    # train the model
    model = btnkm(X_train.shape[1])  
    R, _, _, _, _, _, _= model.train(
        features=X_train,
        target=y_train,
        input_dimension=input_dimension,
        max_rank=max_rank,
        shape_parameter_tau=a,
        scale_parameter_tau=b,
        shape_parameter_lambda_R=c,
        scale_parameter_lambda_R=d,
        shape_parameter_lambda_M=g,
        scale_parameter_lambda_M=h,
        max_iter=50,
        precision_update=True,
        lambda_R_update=True,
        lambda_M_update=True,
        plot_results=False,
        prune_rank=True,
        classification=True
    )
    
    # Predict (mse is returned by the predict function)
    prediction_mean, prediction_std, mse = model.predict(
        features=X_test,
        input_dimension=input_dimension,
        true_values=y_test,
        classification =True
    )

    mse_values.append(mse)
    R_effective.append(R)

    #NLL 
    probs_gt_zero = norm.sf(0, loc=prediction_mean, scale=prediction_std)  # P(y > 0)
    y_test_binary = (y_test + 1) // 2
    eps = 1e-15
    y_pred_prob = np.clip(probs_gt_zero, eps, 1 - eps)
    nll = -np.mean(y_test_binary * np.log(y_pred_prob) + (1 - y_test_binary) * np.log(1 - y_pred_prob))
    nll_values.append(nll)


end_time = time.time()  


total_runtime_seconds = end_time - start_time
print(f"Total runtime for 10 runs: {total_runtime_seconds:.2f} seconds")


mean_mse, std_mse = np.mean(mse_values), np.std(mse_values)
effective_r, effective_r_std = np.mean(R_effective), np.std(R_effective)
mean_nll, std_nll = np.mean(nll_values), np.std(nll_values)


print(f"Mean Missclassication: {mean_mse}, Standard Deviation of Missclassication: {std_mse}")
print(f"Mean Effective R: {effective_r}, std: {effective_r_std}")
print(f"Mean NLL: {mean_nll}, std_nll: {std_nll} ")


# REPORTED

# input_dimension = 30
# # max_rank = 25
# Total runtime for 10 runs: 3785.66 seconds
# Mean Missclassication: 0.07462039045553145, Standard Deviation of Missclassication: 0.014718607768926126
# Mean Effective R: 8.6, std: 1.2
# Mean NLL: 0.4991748394689689, std_nll: 0.04722475764046458 