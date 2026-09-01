# PrivGen — Differentially Private Retrieval-Augmented Diffusion

PrivGen implements **Differentially Private (DP) Retrieval-Augmented Diffusion Models (RDM)** for image generation. It provides provable $(\epsilon, \delta)$-Differential Privacy guarantees when querying private reference image datasets by combining MetaCLIP semantic retrieval with Rényi Differential Privacy (RDP) accounting.

## Privacy Mechanics

- **No Private Fine-Tuning Required**: Base diffusion models remain frozen; private information enters solely through retrieval queries protected by noisy aggregation.
- **Rényi Differential Privacy (RDP) Accounting**: Accurately tracks cumulative privacy expenditure across multi-step diffusion sampling.
- **MetaCLIP Semantic Retrieval**: High-fidelity cross-modal text-to-image neighbor matching with face-blurring and privacy filters.

## Usage

```bash
# Run private sample generation with DP privacy budget tracking
python main.py --config configs/in64fb_sig0.05_agg.yaml
```
