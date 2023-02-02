import joblib

def predict(data):
    regression = joblib.load('regressionModel')
    return regression.predict(data)