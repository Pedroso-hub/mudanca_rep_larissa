import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Used for join all results from different training evaluations
def join_eval(save_file:str, dir:str):
    arr = os.listdir(dir)
    results = []
    for file in arr:
        df = pd.read_csv(dir + '/' + file)
        #     for item, row in df.iterrows():
        results.append({
        'file': file,
        'epoch': df.tail(1)['epoch'].values[0],
        'val_CCC': df.tail(1)['val_CCC'].values[0],
        'val_loss': df.tail(1)['val_loss'].values[0],
        'val_root_mean_squared_error': df.tail(1)['val_root_mean_squared_error'].values[0],   
        })
            
    results = sorted(results, key=lambda k: k['val_CCC'], reverse=False)
    eval = pd.DataFrame(results)
    eval.to_csv(save_file)

# eval('results.csv',"./result/feat",)

def ccc(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc


def bimodal():
    audio = pd.read_csv('./result/prediction/feat_pred.csv')
    texto = pd.read_csv('./result/prediction/all-mpnet-base-v2_pred.csv')
    audio['v_text'] = texto['v'] #add the price_2 column from df2 to df1
    audio['a_text'] = texto['a']
    audio['d_text'] = texto['d']
    audio['v_mean'] = ((audio['v_text'] + audio['v']) / 2)
    audio['a_mean'] = ((audio['a_text'] + audio['a']) / 2)
    audio['d_mean'] = ((audio['d_text'] + audio['d']) / 2)
    y_test  = np.load('./data_iemocap/feat/y_Test.npy')
    gold_list = []
    for item in y_test:
        gold_list.append({'v': item[0], 'a': item[1], 'd':item[2] })

    df_gold = pd.DataFrame(gold_list, columns=['v', 'a', 'd'])
    result = {
        'ccc_v': round(ccc(df_gold['v'], audio['v_mean']), 4),
        'ccc_a': round(ccc(df_gold['a'], audio['a_mean']), 4),
        'ccc_d': round(ccc (df_gold['d'], audio['d_mean']), 4),
        'mse_v': round(mean_squared_error(df_gold['v'], audio['v_mean']), 4),
        'mse_a': round(mean_squared_error(df_gold['a'], audio['a_mean']), 4),
        'mse_d': round(mean_squared_error(df_gold['d'], audio['d_mean']), 4),
        }
    print('baseline', result)

    result = {
        'ccc_v': round(ccc(df_gold['v'], audio['v']), 4),
        'ccc_a': round(ccc(df_gold['a'], audio['a']), 4),
        'ccc_d': round(ccc(df_gold['d'], audio['d']), 4),
        'mse_v': round(mean_squared_error(df_gold['v'], audio['v']), 4),
        'mse_a': round(mean_squared_error(df_gold['a'], audio['a']), 4),
        'mse_d': round(mean_squared_error(df_gold['d'], audio['d']), 4),
        }
    print('audio only', result)
    result = {
        'ccc_v': round(ccc(df_gold['v'], audio['v_text']), 4),
        'ccc_a': round(ccc(df_gold['a'], audio['a_text']), 4),
        'ccc_d': round(ccc(df_gold['d'], audio['d_text']), 4),
        'mse_v': round(mean_squared_error(df_gold['v'], audio['v_text']), 4),
        'mse_a': round(mean_squared_error(df_gold['a'], audio['a_text']), 4),
        'mse_d': round(mean_squared_error(df_gold['d'], audio['d_text']), 4),
        }
    print('text only', result)

def embedding_with_feat():
    audio_v = pd.read_csv('./result_harpy/prediction/1__vggish_pred.csv')
    audio_ad = pd.read_csv('./result_harpy/prediction/2__ComParE_2016_pred.csv')
    texto_v = pd.read_csv('./result_harpy/prediction/1__paraphrase-MiniLM-L3-v2_pred.csv')

    results = pd.DataFrame(columns=['v', 'a', 'd'])
    results['v'] = ((audio_v['v'] + texto_v['v']) / 2)
    results['a'] = audio_ad['a']
    results['d'] = audio_ad['d']


    y_test  = np.load('./data_iemocap/feat/y_Test.npy')
    gold_list = []
    for item in y_test:
        gold_list.append({'v': item[0], 'a': item[1], 'd':item[2] })

    df_gold = pd.DataFrame(gold_list, columns=['v', 'a', 'd'])
    result = {
        'ccc_v': round(ccc(df_gold['v'], results['v']), 4),
        'ccc_a': round(ccc(df_gold['a'], results['a']), 4),
        'ccc_d': round(ccc (df_gold['d'], results['d']), 4),
        'mse_v': round(mean_squared_error(df_gold['v'], results['v']), 4),
        'mse_a': round(mean_squared_error(df_gold['a'], results['a']), 4),
        'mse_d': round(mean_squared_error(df_gold['d'], results['d']), 4),
        }
    print('baseline', result)

    # print(audio.head())
# bimodal()
