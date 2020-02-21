import unittest
import pandas as pd
from keras.models import load_model
import keras.backend as K

def patent_value_loss(y_true, y_pred):
    '''
    Since this was present during training for the patent count model,
    it must be loaded as a custom object at the time of prediction.
    Remove the custom_object code below if you are predicting on
    citation counts (measure of quality).
    '''
    patent_value_loss = K.abs(1 - K.exp(y_true - y_pred)) * 50000
    return patent_value_loss

def pred_data(i, j):
    '''
    Loads the data, converts to array, transforms the data, returns it
    '''
    data = pd.read_csv('test_data.csv')
    data = data[i:j]
    scalerfile = 'Patent_text_cosine_similarity_training_MinMaxScaler-2-14-20.save'
    scaler = joblib.load(open(scalerfile, 'rb'))
    # Transforming sample data
    # (sample.reshape() is required because it is just one line, an array)
    X = scaler.transform(data.reshape(-1, 31))
    return X

def predict_patent(X):
    '''
    Hard-coded for the patent count model weights in github
    Loads model, predicts on X, and returns prediction
    '''
    model = "patent_text_model_epoch_no.030-2-14-20.h5"
    patent_model = load_model(model,
                              custom_objects={'patent_value_loss': patent_value_loss})
    prediction = patent_model.predict(X)
    return prediction

class patentTest(unittest.TestCase):
    def test(self):
        self.assertEqual(predict_patent(X=pred_data(0,1)), -0.0048939) # 0 + bias
        self.assertEqual(predict_patent(X=pred_data(1,2)), 1.1016113)
        self.assertEqual(predict_patent(X=pred_data(2,3)), 1.682951)
        self.assertEqual(predict_patent(X=pred_data(3,4)), 1.3852881)
        self.assertEqual(predict_patent(X=pred_data(4,5)), 0.29458386)
