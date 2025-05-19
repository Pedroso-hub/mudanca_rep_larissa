# Import TF 2.X and make sure we're running eager.
import argparse
import tensorflow.compat.v2 as tf
import sys
sys.path.append('..') 
tf.enable_v2_behavior()
assert tf.executing_eagerly()
import ffmpeg
import tensorflow_hub as hub
import numpy as np
from utils import load_audio
import pandas as pd
import time
# from essentia.standard import TensorflowPredict2D
import os 
from join import pad_or_trim

# Load the module and run inference. TRILL
def generate_embedding_TRILL(audio:str):
    module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3')
    # `wav_as_float_or_int16` can be a numpy array or tf.Tensor of float type ors
    # int16. The sample rate must be 16kHz. Resample to this sample rate, if
    # necessary.
    wav_as_float_or_int16 = load_audio(audio)
    emb = module(samples=wav_as_float_or_int16, sample_rate=16000)['embedding']
    # `emb` is a [time, feature_dim] Tensor.
    # emb.shape.assert_is_compatible_with([128, 100])
    emb = emb.numpy()
    return emb

def generate_embedding_VGGISH(audio:str):
   
    # Input: 3 seconds of silence as mono 16 kHz waveform samples.
    waveform = load_audio(audio)
    waveform = pad_or_trim(waveform)
    # Run the model, check the output.
    embeddings = model(waveform)
    embeddings.shape.assert_is_compatible_with([None, 128])
    
    # print(embeddings)
    return embeddings.numpy()

def create_dataset(embedding:str):
    dataset_dir = '../../datasets/IEMOCAP_full_release/'
    df_iemocap = pd.read_csv('../../datasets/df_iemocap_eval_full_splited.csv')
    df_group2 = df_iemocap.groupby(['split'])
    
    for group_name, df_group in df_group2:
        X = []
        golden_stand = []
        print(df_group.shape[0])
        data_len = df_group.shape[0]
        start_time = time.time()

        for index, row in df_group.iterrows():
            golden_stand.append([row['val'], row['act'], row['dom']])
            wav_file = dataset_dir + row['dir'] + '/' + row['wav_file'] + '.wav'
                # (returns signal as a numpy array)
            if embedding == 'trill':
                em = generate_embedding_TRILL(wav_file)
            elif embedding == 'vggish':
                em = generate_embedding_VGGISH(wav_file)

            X.append(em)
            
        if embedding == 'trill':
            X_avg = np.zeros((data_len, 2048))
        else:
            X_avg = np.zeros((data_len, 384))
        # trill
        # 
        for index, embd in enumerate(X):
            X_avg[index] = np.average(embd, axis=0)
            # X_avg[index] = embd.flatten()

        print(group_name)
        file_name_y = args.path + embedding + '/y_' + group_name
        file_name_x = args.path + embedding + '/x_' + group_name
        np.save(file_name_y, np.array(golden_stand))
        np.save(file_name_x, X_avg)
        print('time: ', round((time.time() - start_time),4),)
        with open(args.path + embedding + '/' + group_name + '_time.txt', 'w') as f:
            f.write(str(round((time.time() - start_time),4)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model = hub.load('https://tfhub.dev/google/vggish/1')
    # parser.add_argument("dataframe", type=str, help="Path to dataframe containing gold standard annotation.")
    parser.add_argument("-model", type=str, help="Embedding model name.")
    parser.add_argument("-path", type=str, help="Dir path to save data")
    args = parser.parse_args()

    if not os.path.exists(args.path+args.model): 
        os.makedirs(args.path+args.model) 

    create_dataset(args.model)