#!/bin/bash
# v24 queue: waits for the data rebuild, scores v22 / v23 on the cleaned test set, trains the three finetune variants (+hp, +gold, +hp and gold),
# then the full alpha + beta pretraining with hp and gold. Models run one at a time (the experiments file is not safe to append to in parallel).
# After each model it is logged in model_objects/experiments.csv by train_model.py and outstanding_experiments.md is refreshed by v24_record.py.
# Progress lines start with QUEUE so they can be watched. Run from the project root.
cd "$(dirname "$0")/.."
PY=python3
D=training_data_v24
COMMON="data_dir_name=$D upgrade_encoding=cumulative upgrade_flip_weight=1 dropout=0.1 lr=0.03 lr_schedule=cosine lr_final_factor=0.01 batch_size=64 select_metric=log_loss"

echo "QUEUE waiting for data rebuild"
while pgrep -f "scripts/process_data.py" > /dev/null; do sleep 30; done
for d in finetuning pre_training_beta pre_training_alpha; do
  for s in train test; do
    if [ ! -s data/$D/$d/$s/$s.bin ]; then echo "QUEUE ABORT missing data/$D/$d/$s/$s.bin"; exit 1; fi
  done
done
echo "QUEUE data ready"

$PY scripts/eval_models.py $D model_objects/bench/v24/baselines.csv fnn_v22_ft_mt_cum fnn_v22_ft_mt_cum_flip1 fnn_v22_alpha_beta_mt_cum > /dev/null 2>&1 \
  && echo "QUEUE DONE baselines" || echo "QUEUE FAILED baselines"
$PY scripts/v24_record.py > /dev/null 2>&1

run() { # name, extra args
  name=$1; shift
  echo "QUEUE START $name $(date +%H:%M)"
  if $PY scripts/train_model.py --set model_version=$name $COMMON "$@" > model_objects/dev_logs_$name.txt 2>&1; then
    echo "QUEUE DONE $name $(date +%H:%M)"
  else
    echo "QUEUE FAILED $name $(date +%H:%M)"
  fi
  $PY scripts/v24_record.py > /dev/null 2>&1
}

run fnn_v24_ft_mt_hp 'extra_features=["hp"]' 'stages=["finetuning"]'
run fnn_v24_ft_mt_gold 'extra_features=["gold"]' 'stages=["finetuning"]'
run fnn_v24_ft_mt_hp_gold 'extra_features=["hp","gold"]' 'stages=["finetuning"]'
run fnn_v24_alpha_beta_mt_hp_gold 'extra_features=["hp","gold"]' 'stages=["pre_training_alpha","pre_training_beta","finetuning"]' 'epochs={"pre_training_alpha":4,"pre_training_beta":10,"finetuning":20}'
echo "QUEUE ALL DONE"
