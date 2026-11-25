# AirFusion

Research code for **reliability-aware multimodal sensor fusion** in civilian airspace safety: classifying an airborne
object as bird-like, small-UAV-like or unknown from passive image-summary, acoustic-summary and telemetry-summary
features, and testing how the classifier behaves when sensors are noisy or missing. It is a classification study only;
it does not control aircraft or act on what it classifies.

**Status:** the data in `data/` are **synthetic**. They exercise the pipeline; they are not evidence about real sensors,
and no research claim is made from them. Real experiments need a licensed, ethically collected dataset.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m airfusion.generate_demo   # writes data/synthetic_sensor_data.csv
python -m airfusion.train           # trains the baseline
python -m airfusion.evaluate        # macro-F1, balanced accuracy, per-class report
python -m pytest -q
```

On the synthetic data the baseline reaches macro-F1 0.93 on 240 test rows; this only shows that the pipeline runs.

## Code

| Module | Role |
|---|---|
| `airfusion/data.py` | Feature definitions and synthetic data generator |
| `airfusion/fusion.py` | One model per sensor, and late fusion that averages the probabilities of the sensors that are available |
| `airfusion/model.py`, `train.py`, `evaluate.py` | Baseline model, training, evaluation |
| `airfusion/robustness.py` | Noise and missing-modality stress tests |

Planned evaluation (see `research/PAPER_OUTLINE.md`): early vs late vs reliability-aware fusion, macro-F1, balanced
accuracy, Brier score and calibration error under modality dropout, with confidence intervals.

## How to cite

```bibtex
@software{hussain2026airfusion,
  author = {Hussain, Muzammal},
  title  = {{AirFusion: reliability-aware multimodal sensor fusion for civilian airspace safety}},
  year   = {2026},
  url    = {https://github.com/MHKhan50/airspace-sensor-fusion-robustness}
}
```

## Author

Muzammal Hussain, independent researcher, Glasgow, United Kingdom.
ORCID: [0009-0001-2570-5686](https://orcid.org/0009-0001-2570-5686)

## Licence

MIT. See `LICENSE`.

