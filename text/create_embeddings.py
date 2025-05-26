import os
import pandas as pd
import time
import argparse
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

import numpy as np
import logging
logging.getLogger().setLevel(logging.ERROR)
#   'mpnet': 'sentence-transformers/all-mpnet-base-v2',
#                     'minilml3': 'sentence-transformers/paraphrase-MiniLM-L3-v2',
#                     'minilml12': 'sentence-transformers/all-MiniLM-L12-v2'


def create_dataset(embedding:str):
    model = SentenceTransformer(embedding)
    print("entrou na create_dataset")
    oi = os.path.normpath(os.getcwd() + os.sep + os.pardir+os.sep + os.pardir+os.sep + os.pardir)
    dataset_dir = oi+'\\datasets\\IEMOCAP_full_release\\'
    df_iemocap = pd.read_csv(oi+'\\datasets\\df_iemocap_eval_full_splited.csv')

    df_group2 = df_iemocap.groupby(['split'])
    
    for group_name, df_group in df_group2:
        X = []
        golden_stand = []
        data_len = df_group.shape[0]
        references = []
        # if group_name != 'Development' and group_name != 'Test':
        start_time = time.time()
        # if group_name == 'Test':
        for index, row in df_group.iterrows():
            
            path = row['dir'].split("/")
            full_path = dataset_dir + path[0] + '/dialog/transcriptions/' + path[-1] + '.txt'
            golden_stand.append([row['val'], row['act'], row['dom']])
            with open(full_path) as f:
                contents = f.readlines()
                for item in contents:
                    txt_file = item.split(" [")
                    txt_file = txt_file[0]
                    if row['wav_file'] == txt_file:
                        embeddings = model.encode(item.split(": ")[-1].strip())
                        # print(embeddings.shape)
                        references.append(embeddings)

        print('time: ', round((time.time() - start_time),4), "group ", group_name, 'embedding', embedding)
        with open('../data_iemocap_sa/' + embedding + '/' + group_name + '_time.txt', 'w') as f:
            f.write(str(round((time.time() - start_time),4)))
        file_name_y = '../data_iemocap_sa/' + embedding + '/y_' + group_name
        file_name_x = '../data_iemocap_sa/' + embedding + '/x_' + group_name
        np.save(file_name_y, np.array(golden_stand))
        np.save(file_name_x, np.array(references))


        
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("dataframe", type=str, help="Path to dataframe containing gold standard annotation.")
    # parser.add_argument("-model", type=str, help="Embedding model name.")
    # args = parser.parse_args()
    # print(args)
    # model = 'all-mpnet-base-v2'
    # if not os.path.exists('../data_iemocap_sa/'+ model): 
    #     os.makedirs('../data_iemocap_sa/'+ model) 
    # create_dataset(model)

    model = 'paraphrase-MiniLM-L3-v2'

    if not os.path.exists('../data_iemocap/paraphrase-MiniLM-L3-v2'): 
        os.makedirs('../data_iemocap/paraphrase-MiniLM-L3-v2') 
    create_dataset(model)

