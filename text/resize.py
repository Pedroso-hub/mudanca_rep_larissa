import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import time

from sklearn.decomposition import PCA

# import umap.umap_ as UMAP

def standardization(feat):
    scaler = StandardScaler()
    scaler = scaler.fit(feat)
    scaled_feat = scaler.transform(feat)
    return scaled_feat

def data_size_scaling(algorithm, data):
    start_time = time.time()
    reduced_data = algorithm.fit_transform(standardization(data))
    elapsed_time = time.time() - start_time
    print('runtime (s) :' + str(elapsed_time))
    return np.asarray(reduced_data)
    # del subsample
    # result.append((size, elapsed_time))
    # return pd.DataFrame(result, columns=('dataset size', 'runtime (s)'))

n_components = 128
all_algorithms = [
    PCA(n_components=n_components),
]

embeddings_path = '../data_iemocap_sa/paraphrase-MiniLM-L3-v2/'

# add to x_Development.npy, x_Test.npy, x_Train.npy
x_Development = np.load(embeddings_path + "x_Development.npy")
x_Test = np.load(embeddings_path + "x_Test.npy")
x_Train = np.load(embeddings_path + "x_Train.npy")

performance_data = {}
for algorithm in all_algorithms:

    alg_name = str(algorithm).split('(')[0]
    
    # performance_data[alg_name] = data_size_scaling(algorithm, mnist_data, n_runs=5)
    print(f"[{time.asctime(time.localtime())}] Start {alg_name}")
    
    np.save(embeddings_path + "x_Development_" + alg_name, np.array(data_size_scaling(algorithm, x_Development)))
    np.save(embeddings_path + "x_Test_" + alg_name, np.array(data_size_scaling(algorithm, x_Test)))
    np.save(embeddings_path + "x_Train_" + alg_name, np.array(data_size_scaling(algorithm, x_Train)))

    print(f"[{time.asctime(time.localtime())}] Completed {alg_name}")