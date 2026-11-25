import numpy as np
import pandas as pd
from pathlib import Path

FEATURES = ["rgb_shape","rgb_motion","rgb_brightness","audio_low","audio_high","audio_periodicity","telemetry_speed","telemetry_altitude","rgb_available","audio_available","telemetry_available"]
LABELS = ["bird_like","small_uav_like","unknown_object"]

def make_data(n=1200, seed=42):
    rng=np.random.default_rng(seed); y=rng.integers(0,3,n)
    centers=np.array([[.25,.35,.6,.65,.2,.25,.25,.4],[.7,.65,.45,.2,.75,.7,.65,.6],[.45,.5,.5,.45,.45,.45,.45,.5]])
    x=np.clip(centers[y]+rng.normal(0,.16,(n,8)),0,1)
    avail=(rng.random((n,3))>.08).astype(float)
    x[:,:3]*=avail[:,[0]]; x[:,3:6]*=avail[:,[1]]; x[:,6:8]*=avail[:,[2]]
    df=pd.DataFrame(x,columns=FEATURES[:8]); df[FEATURES[8:]]=avail
    df["label"]=[LABELS[i] for i in y]
    return df

def save(path="data/synthetic_sensor_data.csv", n=1200, seed=42):
    Path(path).parent.mkdir(parents=True,exist_ok=True); make_data(n,seed).to_csv(path,index=False); return path
