from math import log
import numpy as np
from sigmoid import sigmoid

def cost_function(X,y,theta):

    M = X.shape[0]
    hypothese = sigmoid(X@theta)
    cost = (-y*np.log(hypothese))-((1-y)*np.log(1-hypothese))
    J = (1/2*M)* sum(cost)

    return float(J)