from numpy import load as np_load
from joblib import dump, load
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
import sys
from numpy import unique
import os
import struct

struct_fmt = 'i' + 'f' * 6  # int[5], float, byte[255]
struct_len = struct.calcsize(struct_fmt)
struct_unpack = struct.Struct(struct_fmt).unpack_from
results = []
train_data = []

files = os.listdir('./raw_data')

for file in files:
    with open(f"./raw_data/{file}", "rb") as f1:
        while True:
            data = f1.read(28)
            if not data:
                break
            s = list(struct_unpack(data))
            train_data.append(s)

train_data = unique(train_data, axis=0)


# savez_compressed('train_data.npz', train_data)


def save_model(clf):
    model_name = type(clf).__name__
    dump(clf, f"./models/{model_name}.joblib")


def load_model(model_file_name):
    clf = load(model_file_name)
    return clf


def initialize_train_model(model_name, X_train, y_train):
    model_ft = None
    if model_name == "sgd":
        """ SGDClassifier
        """
        model_ft = SGDClassifier(loss='modified_huber', shuffle=True, random_state=101)
        model_ft.fit(X_train, y_train)
    if model_name == "knn":
        """ KNeighborsClassifier
        """
        model_ft = KNeighborsClassifier(n_neighbors=15)
        model_ft.fit(X_train, y_train)
    if model_name == "logistic_regression":
        """ LogisticRegression
        """
        model_ft = LogisticRegression(solver='lbfgs', max_iter=1000)
        model_ft.fit(X_train, y_train)
    if model_name == "decision_tree":
        """ DecisionTreeClassifier
        """
        model_ft = DecisionTreeClassifier(max_depth=10, random_state=101,
                                          max_features=None, min_samples_leaf=15)
        model_ft.fit(X_train, y_train)
    if model_name == "random_forest":
        """RandomForestClassifier
        """
        model_ft = RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=1,
                                          random_state=101, max_features=None, min_samples_leaf=30)
        model_ft.fit(X_train, y_train)
    if model_name == "svm":
        """SVM
        """
        model_ft = SVC(kernel="linear", C=0.025, random_state=101)
        model_ft.fit(X_train, y_train)
    if model_name == "naive_bayes":
        """Naive Bayes
        """
        model_ft = GaussianNB()
        model_ft.fit(X_train, y_train)
    return model_ft


# model_name = sys.argv[1]

# read dataset
# buttons = np_load('train_data.npz')
# # training dataset
X_buttons = train_data[:, 1:]
# # testing dataset
# y_buttons = train_data['arr_0'][:, 0]
y_buttons = train_data[:, 0]
random_value = randint(0, 10)
X_train, X_test, y_train, y_test = train_test_split(X_buttons, y_buttons, test_size=0.25, random_state=random_value)

""" SGDClassifier
"""
model_name = "sgd"
model = initialize_train_model(model_name, X_train, y_train)

# test model
accuracy = model.score(X_test, y_test)
print(f"SGDClassifier: {accuracy}")

# save model
save_model(model)
# load model
# model_name = type(model).__name__
# load_model(f"./models/{model_name}.joblib")

""" KNeighborsClassifier
"""
model_name = "knn"
model = initialize_train_model(model_name, X_train, y_train)

# test model
accuracy = model.score(X_test, y_test)
print(f"KNeighborsClassifier: {accuracy}")

# save model
save_model(model)

""" LogisticRegression
"""
model_name = "logistic_regression"
model = initialize_train_model(model_name, X_train, y_train)

# test model
accuracy = model.score(X_test, y_test)
print(f"LogisticRegression: {accuracy}")

# save model
save_model(model)

""" DecisionTreeClassifier
"""
model_name = "decision_tree"
model = initialize_train_model(model_name, X_train, y_train)

# test model
accuracy = model.score(X_test, y_test)
print(f"DecisionTreeClassifier: {accuracy}")

# save model
save_model(model)

""" RandomForestClassifier
"""
model_name = "random_forest"
model = initialize_train_model(model_name, X_train, y_train)

# test model
accuracy = model.score(X_test, y_test)
print(f"RandomForestClassifier: {accuracy}")

# save model
save_model(model)

""" Naive Bayes
"""
model_name = "naive_bayes"
model = initialize_train_model(model_name, X_train, y_train)

# test model
accuracy = model.score(X_test, y_test)
print(f"Naive Bayes: {accuracy}")

# save model
save_model(model)

""" SGDClassifier
"""
model_name = "sgd"
model = initialize_train_model(model_name, X_train, y_train)

# test model
accuracy = model.score(X_test, y_test)
print(f"SGDClassifier: {accuracy}")

# save model
save_model(model)
