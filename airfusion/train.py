from pathlib import Path
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from .data import FEATURES, save
from .model import build
def main():
    if not Path("data/synthetic_sensor_data.csv").exists(): save()
    d=pd.read_csv("data/synthetic_sensor_data.csv")
    x,_,y,_=train_test_split(d[FEATURES],d.label,test_size=.2,random_state=42,stratify=d.label)
    m=build().fit(x,y); Path("artifacts").mkdir(exist_ok=True); joblib.dump(m,"artifacts/baseline.joblib"); print("Saved baseline")
if __name__=="__main__": main()
