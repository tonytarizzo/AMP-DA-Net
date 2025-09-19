# run_collection.sh

python -u -m scripts.1_dataset_collection \
  --data-dir              "runs/datasets" \
  --save-dir              "runs/results" \
  \
  --train-size             40000 \
  --valid-size             10000 \
  --test-size              2000 \
  --early-stopping-patience 1000 \
  \
  --model                 resnet \
  --K-t                   40 \
  --min-p                 7 \
  --max-p                 13 \
  --frac-random           0.2 \
  \
  --message-split         20 \
  --n                     128 \
  --dim                   64 \
  --code-order            pop \
  \
  --total-rounds          1000 \
  --local-epochs          3 \
  --local-lr              0.01 \
  \
  --optimizer             fedavg \
  --global-lr             1.0 \
  --momentum              0.0 \
  --beta1                 0.9 \
  --beta2                 0.99 \
  --eps                   0 \
  \
  --batch-size-train      16 \
  --batch-size-test       16 \
  \
  --seed                  42
