# run_pretrain.sh

python -u -m scripts.2_pretraining \
  --train-size            256000 \
  --valid-size            32000 \
  --test-size             20000 \
  --num-test-rounds       5 \
  --pt-epochs             500 \
  --pt-batch-size         64 \
  --pt-patience           50 \
  --pt-delta              1e-6 \
  --sched-patience        20 \
  --sched-factor          0.5 \
  \
  --dataset-dir           "runs/datasets/iFed_datasets" \
  --code-order            pop \
  --model                 resnet \
  --frac-random           0.2 \
  --within-round          random \
  \
  --K-t                   40 \
  --min-p                 7 \
  --max-p                 13 \
  \
  --message-split         20 \
  --n                     128 \
  --dim                   64 \
  --snr-mode              range \
  --snr-min-db            0.0 \
  --snr-max-db            20.0 \
  --test-snrs             '20,15,10,5,3,0' \
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
  --codebook-trainable \
  --codebook-init         gaussian \
  \
  --amp-lr                0.0001 \
  --lambda-sparse         0.001 \
  --lambda-w              0.001 \
  --lambda-k              0.01 \
  \
  --seed                  42 \
  --ckpt-dir              "runs/checkpoints" \
  --save-dir              "runs/results"
