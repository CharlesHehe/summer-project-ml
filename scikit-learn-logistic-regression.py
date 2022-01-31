from numpy import load as np_load
from joblib import dump, load
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model_file_name = "logistic-regression.joblib"


def save_model(clf):
    dump(clf, model_file_name)


def load_model():
    clf = load(model_file_name)
    return clf


# read dataset
data = np_load('data.npz')
# training dataset
X = data['arr_0'][:, 1:]
# testing dataset
y = data['arr_0'][:, 0]
random_value = randint(0, 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_value)



# define the model: LogisticRegression
# model = LogisticRegression(random_state=random_value).fit(X_train, y_train)

# assign a cluster to each example
# y_prediction = model.predict(X_test)
# model.predict_proba(X_test)


model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
model.fit(X_train, y_train)

# gnb = GaussianNB()
# model = gnb.fit(X_train, y_train)
# test model
accuracy = model.score(X_test, y_test)
print(accuracy)
# save model
save_model(model)
# load model
model = load_model()
