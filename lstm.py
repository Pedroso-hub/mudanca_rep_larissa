
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
import sys
sys.path.append('..') 
import argparse
import numpy as np
from keras.layers import LSTM, Input, Concatenate, Average, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.metrics import MeanSquaredError
from keras.callbacks import CSVLogger
import pandas as pd
from compute_results import ccc
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time

def standardization(feat):
    scaler = StandardScaler()
    scaler = scaler.fit(feat)
    scaled_feat = scaler.transform(feat)
    return scaled_feat

def load_data(dir:str, dimen:int, use_pca: bool):
    X_train = np.load(dir + 'x_Train.npy')
    y_train = np.load(dir + 'y_Train.npy')
    X_test  = np.load(dir + 'x_Test.npy')
    y_test  = np.load(dir + 'y_Test.npy')
    X_dev   = np.load(dir + 'x_Development.npy')
    y_dev   = np.load(dir + 'y_Development.npy')

    X_train = standardization(X_train)
    X_test = standardization(X_test)
    X_dev = standardization(X_dev)

    return X_train, y_train, X_test, y_test, X_dev, y_dev


def train_model_functional(X_train_text, X_train_audio, y_train, X_dev_text, X_dev_audio, 
                           y_dev, X_test_text, X_test_audio, y_test, args):
    
    text_input = Input(shape=(X_train_text.shape[1],1), batch_size=args.batch_size)
    # [384]
    audio_input = Input(shape=(X_train_audio.shape[1],1), batch_size=args.batch_size)
 
    if args.fusion == 'concatenate': 
        embeddings = Concatenate()([text_input, audio_input])
    elif args.fusion == 'average':
        embeddings = Average()([text_input, audio_input])

    normalization = BatchNormalization()(embeddings)
    
    lstm_1 = LSTM(args.units, return_sequences=True, activation=args.activation)(normalization)
    lstm_2 = LSTM(256, return_sequences=False)(lstm_1)
    # lstm_3 = LSTM(128, return_sequences=False)(lstm_2)
    dense = Dense(64)(lstm_2)
    # dense_2 = Dense(32)(dense)
    dropout = Dropout(args.dropout)(dense) #Checar outros métodos
    output = Dense(args.dimen, activation=args.activation_output)(dropout)

    model = Model(inputs=[text_input, audio_input], outputs=output)

    #earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)

    if args.optimizer == 'sgd':
        opt = SGD(learning_rate=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=args.learning_rate)
    else:
        opt = Adam(learning_rate=args.learning_rate)

    model.compile(loss=args.loss, optimizer=opt, metrics=[MeanSquaredError()])

    model.summary()
    # model.fit(train_data, validation_data=valid_data, epochs=8, batch_size=10)
    file = "keras-un_" + str(args.units) + "-drop_" + str(args.dropout) + "-lr_" + str(args.learning_rate) + "-opt_" + args.optimizer + "-bat_" + str(args.batch_size) + "-epo_" + str(args.epochs) + "-loss_" + args.loss + "-act_" + args.activation + "-act_out_" + args.activation_output

    csv_logger = CSVLogger('result\\' + args.emb_model + '\\' + file + '.csv')
    # X_train, y_train, X_val, y_val
    model.fit([X_train_text, X_train_audio], y_train, 
              validation_data=([X_dev_text, X_dev_audio], y_dev), 
              validation_batch_size = args.batch_size, 
              epochs=args.epochs, 
              batch_size=args.batch_size,
              callbacks=[csv_logger])
    
    if args.save == 'yes':
        model.save('.\\model\\'+args.model_name + '.h5')
    else:
        start_time = time.time()
        prediction = model.predict([X_test_text, X_test_audio], batch_size=args.batch_size)
        print('time: ', round((time.time() - start_time),4))
        pred_list = []
        gold_list = []
       
        for item in prediction:
            pred_list.append({'v': item[0], 'a': item[1], 'd':item[2] })
        # np.savetxt('prediction.csv',prediction, delimiter=',')
        df_pred = pd.DataFrame(pred_list, columns=['v', 'a', 'd'])

        for item in y_test:
            gold_list.append({'v': item[0], 'a': item[1], 'd':item[2] })

        df_gold = pd.DataFrame(gold_list, columns=['v', 'a', 'd'])
        result = {
            'ccc_v': round(ccc(df_gold['v'], df_pred['v']), 4),
            'ccc_a': round(ccc(df_gold['a'], df_pred['a']), 4),
            'ccc_d': round(ccc(df_gold['d'], df_pred['d']), 4),
            'mse_v': round(mean_squared_error(df_gold['v'], df_pred['v']), 4),
            'mse_a': round(mean_squared_error(df_gold['a'], df_pred['a']), 4),
            'mse_d': round(mean_squared_error(df_gold['d'], df_pred['d']), 4),
        }
        
        print(result)
        
        df_pred.to_csv('.\\result_harpy\\prediction\\'+ str(args.dimen) + '__' + args.model_name + '_pred.csv')
        df_result = pd.DataFrame.from_dict([result])
        df_result.to_csv('.\\result_harpy\\prediction\\'+ str(args.dimen) + '__' + args.model_name + '.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("dataframe", type=str, help="Path to dataframe containing gold standard annotation.")
    parser.add_argument("-units", type=int, help="LSTM units size.")
    parser.add_argument("-dropout", type=float, help="Dropout.")
    parser.add_argument("-learning_rate", type=float, help="Learning rate.")
    parser.add_argument("-optimizer", type=str, help="Optimizer.")
    parser.add_argument("-batch_size", type=int, help="Batch size.")
    parser.add_argument("-epochs", type=int, help="Number of epochs.")
    parser.add_argument("-loss", type=str, help="Loss.")    
    # parser.add_argument("metrics", type=int, help="Number of epochs.")
    parser.add_argument("-activation", type=str, help="Activation function in LSTM.")
    parser.add_argument("-activation_output", type=str, help="Activation function in dense layer")
    # parser.add_argument("data_len", type=int, help="Number of epochs.")
    parser.add_argument("-emb_model", type=str, help="Embedding model")
    parser.add_argument("-save", type=str, help="Create a model file or not")
    parser.add_argument("-model_name", type=str, help="Model file name")
    # parser.add_argument("-x_data", type=str, help="Numpy file with train data")
    # parser.add_argument("-y_data", type=str, help="Numpy file with test data.")
    # parser.add_argument("-z_data", type=str, help="Numpy file with validation data.")
    parser.add_argument("-dir_data", type=str, help="Directory with np data")
    parser.add_argument("-dir_data_text", type=str, help="Directory with np data")
    parser.add_argument("-dimen", type=int, help="Number of dimensions")
    parser.add_argument("-fusion", type=str, help="Embedding fusion type")

    args = parser.parse_args()
   
    X_train_audio, y_train_audio, X_test_audio, y_test_audio, X_dev_audio, y_dev_audio = load_data(args.dir_data, args.dimen, False)

    X_train_text, y_train_text, X_test_text, y_test_text, X_dev_text, y_dev_text = load_data(args.dir_data_text, args.dimen, True)
    y_train = y_train_audio
    y_test = y_test_audio
    y_dev = y_dev_audio

    train_model_functional(X_train_text, X_train_audio, y_train, X_dev_text, X_dev_audio, 
                            y_dev, X_test_text, X_test_audio, y_test, args)
    # 
