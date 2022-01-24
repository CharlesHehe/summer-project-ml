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
x = X_train
# x = array([[0.04, 1.04, 0.04, 0.00, 0.10, 0.00]])
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

response.json()
print(response.json())

# import torch
#
# x = torch.arange(12).view(4, 3)
# print(x, x.stride())
# print(x.is_contiguous())
#
# y = x.t()
# print(y, y.stride())
# print(y.is_contiguous())
#
# y = y.contiguous()
# print(y.stride())
# print(y.is_contiguous())
# y = y.view(-1)
