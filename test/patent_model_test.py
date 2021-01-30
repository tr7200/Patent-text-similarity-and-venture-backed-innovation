import unittest
import pandas as pd
from keras.models import load_model
import keras.backend as K


def patent_value_loss(y_true, y_pred):
    '''custom metric used during training and must be loaded at prediction time'''
    patent_value_loss = K.abs(1 - K.exp(y_true - y_pred)) * 50000
    
    return patent_value_loss

def pred_data(i, j):
    '''Loads the data, converts to array, transforms the data, returns it'''
    DATA = pd.read_csv('test_data.csv')
    data = DATA[i:j]
    SCALERFILE = 'Patent_text_cosine_similarity_training_MinMaxScaler-2-14-20.save'
    scaler = joblib.load(open(SCALERFILE, 'rb'))
    # Transforming sample data
    # (sample.reshape() is required because it is just one line, an array)
    X = scaler.transform(data.reshape(-1, 31))
    
    return X

def predict_patent(X):
    '''Loads model with patent_value_loss, returns prediction on X'''
    MODEL = "patent_text_model_epoch_no.030-2-14-20.h5"
    # Hard-coded for the patent count model weights in github
    patent_model = load_model(MODEL,
                              custom_objects={'patent_value_loss': patent_value_loss})
    prediction = patent_model.predict(X)
    
    return prediction

class patentTest(unittest.TestCase):
    def test1(self):
        self.assertEqual(predict_patent(X=pred_data(0,1)), -0.0048939) # 0 + bias
    def test2(self):
        self.assertEqual(predict_patent(X=pred_data(1,2)), 1.1016113)
    def test3(self):
        self.assertEqual(predict_patent(X=pred_data(2,3)), 1.682951)
    def test4(self):
        self.assertEqual(predict_patent(X=pred_data(3,4)), 1.3852881)
    def test5(self):
        self.assertEqual(predict_patent(X=pred_data(4,5)), 0.29458386)
