import mindspore
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler

# Load data from MAT file
mat_data = sio.loadmat('./MLTSKFS/data/CAL500.mat')
data = mindspore.Tensor(mat_data['data']).float()
target = mindspore.Tensor(mat_data['target']).float()

target[target == -1] = 0

scaler = MinMaxScaler(feature_range=(0, 1))
data = mindspore.Tensor(scaler.fit_transform(data.T).T).float()

# Set optimization parameters
oldOptmParameter = {
    'alpha_searchrange': [0.01, 0.1, 1, 10, 100],
    'beta_searchrange': [0.01, 0.1, 1, 10, 100],
    'gamma_searchrange': [0.01, 0.1, 1, 10, 100],
    'maxIter': 100,
    'minimumLossMargin': 0.01,
    'outputtempresult': 0,
    'drawConvergence': 0,
    'bQuiet': 0
}

TSKoptions = {
    'k_searchrange': [2, 3],
    'h_searchrange': [0.1, 1, 10, 100]
}

BestParameter, BestResult = ML_TSKFS_adaptive_validate(data, target, oldOptmParameter, TSKoptions)

toc = time.time()
elapsed_time = toc - tic
print(f"Elapsed time: {elapsed_time} seconds")
