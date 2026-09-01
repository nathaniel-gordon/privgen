<div align="center">

# 🔒 PrivGen

**Private generative diffusion with formal Rényi Differential Privacy bounds.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Privacy-Preserving%20ML%20%2F%20Diffusion-8b5cf6?style=for-the-badge)](https://github.com/nathaniel-gordon/privgen)

<br/>

*Retrieval-Augmented Diffusion Model (RDM) framework equipped with rigorous Differential Privacy (DP-SGD & RDP accounting). Generates high-fidelity visual samples by querying external knowledge bases without leaking or memorizing individual training examples.*

</div>

---

## 🧠 What Is This?

> **For non-technical readers:** When modern AI generates images, there is a risk that it memorized private photos from its training data (like people's faces, personal medical scans, or proprietary designs) and could reproduce them. PrivGen solves this by using **Differential Privacy** — a mathematical guarantee from cryptography that ensures no individual image in the training database can ever be reconstructed or identified from the model's outputs. You get the quality of modern image generation with provable privacy protection.

---

## 🏗️ Architecture & Privacy Pipeline

PrivGen combines **Retrieval-Augmented Diffusion Models (RDMs)** with **Rényi Differential Privacy (RDP)** accounting and DP-SGD gradient sanitization.

```
🖼️ Private Knowledge Base (Image / Multimodal DB)
                    │
                    ▼
🛡️ Privacy Sanitization & Redaction Layer
   ├── Automated Face Blurring & PII Filtering
   └── MetaCLIP Embedding Vectorization
                    │
                    ▼
⚡ FAISS Accelerated k-NN Index (WebDataset)
   Retrieves Top-K Semantic Neighbors for Conditioning
                    │
                    ▼
🎯 Differentially Private Retrieval-Augmented Diffusion
   ├── Noise Calibrated Gradient Clipping (DP-SGD)
   ├── Cross-Attention Retrieval Conditioning
   └── Continuous Rényi Differential Privacy (RDP) Accumulator
                    │
                    ▼
🎨 High-Fidelity Synthesized Output with Bound (ϵ, δ)-DP Guarantee
```

---

## 🔬 Mathematical Privacy Guarantees

### 1. $(\epsilon, \delta)$-Differential Privacy
A randomized mechanism $\mathcal{M}$ satisfies $(\epsilon, \delta)$-DP if for all neighboring datasets $D, D'$ differing by at most one sample, and all measurable subsets $S \subseteq \text{Range}(\mathcal{M})$:

$$\mathbb{P}[\mathcal{M}(D) \in S] \le e^{\epsilon} \cdot \mathbb{P}[\mathcal{M}(D') \in S] + \delta$$

### 2. Rényi Differential Privacy (RDP) Accounting
PrivGen uses Rényi Differential Privacy of order $\alpha > 1$ to tightly track privacy loss across multi-step diffusion sampling iterations:

$$D_{\alpha}(\mathcal{M}(D) \parallel \mathcal{M}(D')) = \frac{1}{\alpha - 1} \ln \int \left(\frac{p(x)^{\alpha}}{q(x)^{\alpha - 1}}\right) dx \le \rho(\alpha)$$

Under Gaussian mechanism composition with noise multiplier $\sigma$ and subsampling ratio $q$:

$$\rho(\alpha) \approx \frac{\alpha q^2}{2 \sigma^2} \cdot T_{\text{steps}}$$

Converted to standard $(\epsilon, \delta)$-DP via the optimal conversion theorem:

$$\epsilon(\delta) = \min_{\alpha > 1} \left\{ \rho(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1} \right\}$$

---

## 🛠️ Key Technical Capabilities

| Component | Module | Description |
|---|---|---|
| 🔐 **DP-SGD Engine** | `train_private_rdm.py` | Per-sample gradient clipping ($C$) and calibrated Gaussian noise injection ($\sigma$) |
| 📊 **RDP Accountant** | `rdm.util` | Tightly tracks cumulative $(\epsilon, \delta)$ privacy budget across training epochs |
| 🔍 **MetaCLIP Retriever** | `wds_to_clip.py`, `retrieval_utils.py` | Extracts privacy-filtered cross-modal semantic embeddings into FAISS indices |
| 🌊 **Diffusion Core** | `rdm.models.diffusion` | DDPM / DDIM conditioning architectures with cross-attention neighbor injection |
| 📦 **WebDataset FAISS** | `wds_build_faiss.py` | Sharded dataset streaming and fast indexing for multi-million image corpora |

---

## 🚀 Getting Started

```bash
git clone https://github.com/nathaniel-gordon/privgen
cd privgen
pip install -e .
```

### 1. Build the Privacy-Preserving Retrieval Index

```bash
# Convert WebDataset shards to CLIP embeddings
python wds_to_clip.py --input_dir /path/to/shards --output_dir /path/to/embeddings

# Construct the accelerated FAISS search index
python wds_build_faiss.py --embeddings_dir /path/to/embeddings --index_output faiss.index
```

### 2. Train Private Retrieval-Augmented Diffusion Model

```bash
# Launch DP training with calibrated noise sigma=0.05 and target epsilon tracking
python train_private_rdm.py --config configs/in64fb_sig0.05_agg.yaml
```

### 3. Generate Differentially Private Samples

```bash
# Sample from trained private RDM with RDP verification
python main.py --config configs/in64fb_sig0.05_agg.yaml --prompt "A serene forest with sunlight filtering through canopy"
```

---

## 📁 Project Structure

```
privgen/
├── rdm/
│   ├── data/               # Dataset loaders (CIFAR-10, ImageNet, MSCOCO, Adversarial)
│   ├── models/
│   │   ├── diffusion/      # DDPM & DDIM sampling implementations
│   │   └── autoregression/ # Transformer priors
│   ├── modules/
│   │   ├── custom_clip/    # MetaCLIP tokenizer & vision-text encoders
│   │   ├── diffusionmodules/# U-Net openaimodel backbone
│   │   └── retrievers.py   # Privacy-aware k-NN neighbor conditioning
│   └── util.py             # RDP privacy budget calculation & metrics
├── scripts/                # Sharded neighbor search & model training scripts
├── train_private_rdm.py    # Main differentially private training loop
├── wds_build_faiss.py      # FAISS index builder from sharded embeddings
└── main.py                 # Inference & sampling entrypoint
```

---

<div align="center">

*Built by [Nathaniel Gordon](https://github.com/nathaniel-gordon)*

</div>
