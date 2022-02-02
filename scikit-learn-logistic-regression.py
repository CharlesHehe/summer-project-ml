from numpy import load as np_load
from joblib import dump, load
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier


def save_model(clf):
    model_name = type(clf).__name__
    dump(clf, f"{model_name}.joblib")


def load_model(model_file_name):
    clf = load(model_file_name)
    return clf


# read dataset
buttons = np_load('buttons.npz')
# # training dataset
X_buttons = buttons['arr_0'][:, 1:]
# # testing dataset
y_buttons = buttons['arr_0'][:, 0]
random_value = randint(0, 10)
X_train, X_test, y_train, y_test = train_test_split(X_buttons, y_buttons, test_size=0.25, random_state=random_value)

# read dataset
bandsaw = np_load('bandsaw.npz')
# training dataset
X_bandsaw = bandsaw['arr_0'][:, 1:]
# testing dataset
y_bandsaw = bandsaw['arr_0'][:, 0]
random_value = randint(0, 10)
# X_train, X_test, y_train, y_test = train_test_split(X_bandsaw, y_bandsaw, test_size=0.25, random_state=random_value)

# SGDClassifier
# model = SGDClassifier(loss='modified_huber', shuffle=True, random_state=101)
# model.fit(X_train, y_train)
# training:  0.9607462620738323
# testing:  0.8758377668762524


# KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=15)
# model.fit(X_train, y_train)
# training:  0.9880474573280995
# testing:  0.8084571270641885

# LogisticRegression
# model = LogisticRegression(random_state=random_value).fit(X_train, y_train)
# training:  0.9607315602993282
# testing:  0.8758377668762524

# DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=10, random_state=101,
                               max_features=None, min_samples_leaf=15)
model.fit(X_train, y_train)
# training:  0.9999852982254959
# testing:  0.9985213846472742

# Random Forest
# model = RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=1,
#                                random_state=101, max_features=None, min_samples_leaf=30)
# model.fit(X_train, y_train)
# training:  0.9996324556373954
# testing:  0.9985213846472742

# SVM
# model = SVC(kernel="linear", C=0.025, random_state=101)
# model.fit(X_train, y_train)
# training:  0.959673032535027
# testing:  0.8758377668762524

# Naive Bayes
# model = GaussianNB()
# model.fit(X_train, y_train)
# training:  0.554492127199753
# testing:  0.28114419954397846

# test model
accuracy = model.score(X_test, y_test)
print(f"training: ", accuracy)

accuracy = model.score(X_bandsaw, y_bandsaw)
print(f"testing: ", accuracy)
# save model
save_model(model)
# load model
model_name = type(model).__name__
load_model(f"{model_name}.joblib")
