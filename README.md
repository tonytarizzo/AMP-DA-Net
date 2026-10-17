# AMP-DA-Net

Supporting code for **“Learned Digital Over-the-Air Computing for Federated Edge Learning.”**
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

## Conference experiments

The scripts in `recipes/paper/` contain the settings used for the conference
experiments, including the ResNet and VGG training pipelines, the SNR sweep, and
the cross-architecture generalisation experiment.

The tracked `recipes/test_*.sh` scripts provide quick end-to-end checks of the
same pipeline using a smaller model and dataset.


## Citation

If you use AMP-DA-Net or build upon this work, please cite the relevant paper below.

### Conference Paper — IEEE SPAWC 2026

**Learned Digital Over-the-Air Computing for Federated Edge Learning**

Antonio Tarizzo, Mohammad Kazemi, and Deniz Gündüz.

Presented at the *2026 IEEE International Workshop on Signal Processing and Artificial Intelligence for Wireless Communications (SPAWC)*.

[Paper (arXiv)](https://arxiv.org/abs/2509.16577)

```bibtex
@inproceedings{tarizzo2026learneddigital,
  title={Learned Digital Over-the-Air Computing for Federated Edge Learning},
  author={Tarizzo, Antonio and Kazemi, Mohammad and G{\"u}nd{\"u}z, Deniz},
  booktitle={2026 IEEE International Workshop on Signal Processing and Artificial Intelligence for Wireless Communications (SPAWC)},
  year={2026},
  eprint={2509.16577},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2509.16577}
}
```

### Extended Paper — arXiv 2025

**Learned Digital Codes for Over-the-Air Computation in Federated Edge Learning**

Antonio Tarizzo, Mohammad Kazemi, and Deniz Gündüz.

Extended manuscript, arXiv:2512.19777, 2025.

[Paper (arXiv)](https://arxiv.org/abs/2512.19777)

```bibtex
@misc{tarizzo2025learneddigitalcodes,
  title={Learned Digital Codes for Over-the-Air Computation in Federated Edge Learning},
  author={Tarizzo, Antonio and Kazemi, Mohammad and G{\"u}nd{\"u}z, Deniz},
  year={2025},
  eprint={2512.19777},
  archivePrefix={arXiv},
  primaryClass={cs.IT},
  doi={10.48550/arXiv.2512.19777},
  url={https://arxiv.org/abs/2512.19777}
}
```

