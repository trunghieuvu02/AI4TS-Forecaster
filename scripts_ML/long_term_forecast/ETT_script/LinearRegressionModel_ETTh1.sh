model="LinearRegressionModel"
root_path="./dataset/ETT-small"
data_path="ETTh1.csv"
data="ETTh1"
ratio=0.8
lags=96

python -u run_ML.py \
  --task_name long_term_forecast \
  --model $model \
  --root_path $root_path \
  --data_path $data_path \
  --data $data \
  --ratio $ratio \
  --lags $lags \
  --horizon 96 \

python -u run_ML.py \
  --task_name long_term_forecast \
  --model $model \
  --root_path $root_path \
  --data_path $data_path \
  --data $data \
  --ratio $ratio \
  --lags $lags \
  --horizon 192 \

python -u run_ML.py \
  --task_name long_term_forecast \
  --model $model \
  --root_path $root_path \
  --data_path $data_path \
  --data $data \
  --ratio $ratio \
  --lags $lags \
  --horizon 336 \

python -u run_ML.py \
  --task_name long_term_forecast \
  --model $model \
  --root_path $root_path \
  --data_path $data_path \
  --data $data \
  --ratio $ratio \
  --lags $lags \
  --horizon 720 \