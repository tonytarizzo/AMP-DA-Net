# AMP-DA-Net

Supporting code for **“Learned Digital Codes for Over-the-Air Federated Learning” (2025)**.  
This repo contains three end-to-end stages:

1) **Dataset collection** (simulate FL + wireless compression, save iFed bundles)  
2) **Pre-training** the AMP-DA-Net decoder + URA codebook on the collected bundles  
3) **FEEL inference** (federated training with the learned wireless compressor)

---

## Quick start

```bash
# 1) Clone & enter
git clone https://github.com/tonytarizzo/AMP-DA-Net
cd AMP-DA-Net

# 2) Python env
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3) Run the recipes (from repo root)
bash recipes/run_collection.sh
bash recipes/run_pretrain.sh
bash recipes/run_inference.sh

# Or optionally run scripts directly (useful for testing/debugging)
python -m scripts.1_dataset_collection
python -m scripts.2_pretraining
python -m scripts.3_feel_inference
```

The paths for collected datasets and pretrained encoder-decoder pairs are automatically named. This ensures that the next script can locate them automatically, while ensuring the same setup is used for all 3 scripts. If you change one recipe, change the other two to match.

## Citation

If you use this code in your work, please cite (although it is not yet an official cite, will be added to arXiv soon!):

```
@article{tarizzo2025learneddigitalcodesovertheair,
      title={Learned Digital Codes for Over-the-Air Federated Learning}, 
      author={Antonio Tarizzo and Mohammad Kazemi and Deniz Gündüz},
      year={2025},
      eprint={2509.16577},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.16577}, 
}
```
