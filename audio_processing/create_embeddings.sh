# losses=('mse' 'ccc' 'MAE')
models=('vggish')
for model in "${models[@]}"
do
    python embedding_iemocap.py -model $model -path '../eval/data_iemocap_final/'
    # python embedding_msp.py -model $model -path '../eval/data_iemocap_final/'
done