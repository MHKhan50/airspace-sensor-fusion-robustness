import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
from .data import FEATURES, save
from .train import main as train
from pathlib import Path
def main():
    if not Path("data/synthetic_sensor_data.csv").exists(): save()
    if not Path("artifacts/baseline.joblib").exists(): train()
    d=pd.read_csv("data/synthetic_sensor_data.csv"); _,x,_,y=train_test_split(d[FEATURES],d.label,test_size=.2,random_state=42,stratify=d.label)
    p=joblib.load("artifacts/baseline.joblib").predict(x)
    print(classification_report(y,p,zero_division=0)); print("Balanced accuracy:",balanced_accuracy_score(y,p)); print("Macro F1:",f1_score(y,p,average="macro"))
if __name__=="__main__": main()
