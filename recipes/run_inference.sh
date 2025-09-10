# run_inference.sh

python -u -m scripts.3_feel_inference \
  --data-dir              "runs/datasets" \
  --batch-size-train      16 \
  --batch-size-test       16 \
  --frac-random           0.2 \
  \
  --K-t                   40 \
  --min-p                 7 \
  --max-p                 13 \
  --total-rounds          1000 \
  --local-epochs          3 \
  --local-batch-size      20 \
  --local-lr              0.01 \
  --grad-accumulation-size 512 \
  \
  --optimizer             fedavg \
  --global-lr             1.0 \
  --momentum              0.0 \
  --beta1                 0.9 \
  --beta2                 0.99 \
  --eps                   0 \
  \
  --message-split         20 \
  --n                     128 \
  --dim                   64 \
  --snr-db                10.0 \
  --snr-mode              range \
  --snr-min-db            0.0 \
  --snr-max-db            20.0 \
  --post-rounding         \
  --code-order            pop \
  \
  --num-layers            10 \
  --num-filters           32 \
  --kernel-size           3 \
  --use-poisson \
  --update-alpha \
  --learn-sigma2 \
  --finetune-K \
  --finetune-pi \
  --finetune-sigma2 \
  --per-sample-sigma \
  --no-per-sample-alpha \
  --K-max                 -1 \
  --blend-init            0.85 \
  \
  --ckpt-dir              "runs/checkpoints" \
  --codebook-init         gaussian \
  --codebook-trainable \
  --within-round          random \
  --model                 resnet \
  \
  --early-stopping-patience 50 \
  --save-dir              "runs/results" \
  --seed                  42
