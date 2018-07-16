import numpy as np
import sys 

def stdin_reader():
    """ stdin generator reader """
    readline = sys.stdin.readline().strip().split()
    while readline:
        yield readline
        readline = sys.stdin.readline().strip().split()

def svm_sgd(X, y, eta=.5, epochs=10):
    w = np.zeros(X.shape[1])
    for epoch in range(1, epochs):
        for i, x in enumerate(X):
            if (y[i]*np.dot(X[i], w)) < 1:
                w = w + eta * ( (X[i] * y[i]) + (-2  *(1/epoch)* w) )
            else:
                w = w + eta * (-2  *(1/epoch)* w)
    return w


# read
n, m = (int(x) for x in sys.stdin.readline().strip().split())
X = np.zeros((n, m))
y = np.zeros(n)
i = 0
for line in stdin_reader():
    X[i], y[i] = [float(x) for x in line[:-1]], int(line[-1])
    i += 1

# predict 
w = svm_sgd(X, y)
print(' '.join(map(str, w.tolist())))
