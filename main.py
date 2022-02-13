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
        model_ft = LogisticRegression(random_state=random_value)
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


model_name = sys.argv[1]

# read dataset
buttons = np_load('train_data.npz')
# # training dataset
X_buttons = buttons['arr_0'][:, 1:]
# # testing dataset
y_buttons = buttons['arr_0'][:, 0]
random_value = randint(0, 10)
X_train, X_test, y_train, y_test = train_test_split(X_buttons, y_buttons, test_size=0.25, random_state=random_value)

model = initialize_train_model(model_name, X_train, y_train)

# test model
accuracy = model.score(X_test, y_test)
print(f"training: {accuracy}")

# save model
save_model(model)
# load model
model_name = type(model).__name__
load_model(f"./models/{model_name}.joblib")
