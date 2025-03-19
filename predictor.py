
import pickle
from transformers import DateTransformer, SubUrbMeanEncoder


class Predictor:
    def __init__(self):
        with open('model_pipeline.pkl', 'rb') as f:
            self.pipeline = pickle.load(f)

    def predict(self, data):
        prediction = self.pipeline.predict(data)
        return prediction
