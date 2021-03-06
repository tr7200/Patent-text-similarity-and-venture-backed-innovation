import unittest
import pandas as pd
from keras.models import load_model

def pred_data(i, j):
    '''Loads the data, converts to array, transforms the data, returns it'''
    data = pd.read_csv('test_data.csv')
    data = data[i:j]
    scalerfile = 'Citations_text_cosine_similarity_training_MinMaxScaler-2-17-20.save'
    scaler = joblib.load(open(scalerfile, 'rb'))
    # Transforming sample data
    # (sample.reshape() is required because it is just one line, an array)
    X = scaler.transform(data.reshape(-1, 31))
    
    return X

def predict_patent(X):
    '''Loads model, predicts on X, and returns prediction'''
    model = 'citations_text_model_epoch_no.029-2-17-20.h5'
    # Hard-coded for the patent count model weights in github
    patent_model = load_model(model)
    prediction = patent_model.predict(X)
    
    return prediction

class patentTest(unittest.TestCase):
    def test1(self):
        self.assertEqual(predict_patent(X=pred_data(0,1)), 1.0519466)
    def test2(self):
        self.assertEqual(predict_patent(X=pred_data(1,2)), 1.3384985)
    def test3(self):
        self.assertEqual(predict_patent(X=pred_data(2,3)), 1.192138)
    def test4(self):
        self.assertEqual(predict_patent(X=pred_data(3,4)), 1.6698763)
    def test5(self):
        self.assertEqual(predict_patent(X=pred_data(4,5)), 0.36851412)

if __name__ == '__main__':
    unittest.main()
