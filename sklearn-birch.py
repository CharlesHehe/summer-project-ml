from sklearn.cluster import Birch
from numpy import unique
from numpy import where
from numpy import genfromtxt
from matplotlib import pyplot
from joblib import dump, load


def save_model(clf):
    dump(clf, 'model.joblib')


def load_model():
    clf = load('model.joblib')
    return clf


# define dataset
X = genfromtxt("button-delta-coordinates.txt", delimiter=",")
# define the model: Birch
model = Birch(threshold=0.01, n_clusters=3)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
pyplot.rcParams["figure.figsize"] = [7.00, 3.50]
pyplot.rcParams["figure.autolayout"] = True
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    ax.scatter(X[row_ix, 0], X[row_ix, 1], X[row_ix, 2])
# show the plot
pyplot.show()
# save model
save_model(model)
# load model
model = load_model()
# model prediction
model.predict([[10, 10, 19]])
