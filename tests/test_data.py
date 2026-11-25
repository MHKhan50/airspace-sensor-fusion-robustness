from airfusion.data import make_data, FEATURES
def test_data():
    d=make_data(20,1); assert len(d)==20 and all(c in d for c in FEATURES+["label"])
