# SkySentinel AI
Explainable, uncertainty-aware multimodal classification research prototype for **civilian airspace safety**. This starter uses synthetic data only; it is not a real-world drone controller or targeting tool.

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m skysentinel.generate_demo
python -m skysentinel.train
python -m skysentinel.evaluate
```

Research focus: test how passive RGB-like, acoustic-summary and telemetry-summary features behave under noise and missing modalities. Synthetic results are illustrative, not publishable evidence. Replace them with licensed, ethically collected data before making research claims.
