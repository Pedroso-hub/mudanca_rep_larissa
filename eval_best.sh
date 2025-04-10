# losses=('mse' 'ccc' 'MAE')
losses=('mse')
for loss in "${losses[@]}"
do

    python lstm.py -units 128 -dropout 0.25 -learning_rate 0.001 -optimizer 'adam' -batch_size 64 -epochs 50 -activation 'tanh' -activation_output 'tanh' -loss $loss -dir_data './data_msp/vggish/' -emb_model='vggish' -save 'no' -model_name 'vggish' -dimen 3

done
