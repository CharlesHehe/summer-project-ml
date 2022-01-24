from numpy import genfromtxt
from numpy import unique
from numpy import savez_compressed

X = genfromtxt("raw.txt", delimiter=",")
print(X.shape)
X = unique(X, axis=0)

savez_compressed('data.npz', X)