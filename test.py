import requests
from numpy import load as np_load
from numpy import array
from sklearn.model_selection import train_test_split
from random import randint

# read dataset
data = np_load('data.npz')
# training dataset
X = data['arr_0'][:, 1:]
# testing dataset
y = data['arr_0'][:, 0]
random_value = randint(0, 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_value)
# x = X_train

x = array([[0.0801122784614563, 0.9424072504043579, 0.0826762318611145, 0.0, 0.0, 0.0]])
inference_request = {
    "inputs": [
        {
            "name": "predict",
            "shape": x.shape,
            "datatype": "FP32",
            "data": x.tolist()
        }
    ]
}

endpoint = "http://localhost:8080/v2/models/logistic-regression/versions/v0.1.0/infer"
response = requests.post(endpoint, json=inference_request)

print(response.json())
