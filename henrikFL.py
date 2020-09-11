"""Helper functions for my FL+CoMAC simulations"""
import numpy as np

def normalize_weights(weight_updates):
    #Calculate mean and variance
    mean = 0
    variance = 0
    num_devices = len(weight_updates)
    for d in range(num_devices):
        #Collapse weight datastructure into array
        coll_weights = 0
        num_layers = len(weight_updates[d])
        for l in range(num_layers):
            if l == 0:
                coll_weights = weight_updates[d][l].flatten()
            else:
                coll_weights = np.append(coll_weights, weight_updates[d][l].flatten())
        #Collapse complete
        mean = mean + np.mean(coll_weights)
        variance = variance + np.var(coll_weights)
    mean = mean/num_devices
    variance = variance/num_devices
    #Normalize weight updates
    for d in range(num_devices):
        for l in range(num_layers):
            for weight in np.nditer(weight_updates[d][l], op_flags = ['readwrite']):
                weight[...] = (weight-mean)/np.sqrt(variance)
    return weight_updates, mean, variance