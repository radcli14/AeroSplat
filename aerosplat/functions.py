import numpy as np
import random


"""Creates a random quaternion with unit magnitude as a NumPy array"""
def random_unit_quaternion():
    random_quat = np.random.normal(size=4)
    random_quat /= np.linalg.norm(random_quat)
    return random_quat


"""Takes a NumPy array with four coordinates and normalizes it for usage as an orientation quaternion"""
def normalize_quaternion(quaternion: np.array):
     return quaternion / np.linalg.norm(quaternion)
     

"""Gets a length scale from the norm of the length, height, depth dimensions of a volume domain"""
def length_scale(domain: np.ndarray):
     return np.linalg.norm([dimension[1] - dimension[0] for dimension in domain])


"""Creates coordinates for a random point in the provided volume domain, in two or three dimension"""
def point_at_random(domain: np.ndarray):
        x = random.uniform(*domain[0])
        y = random.uniform(*domain[1])
        if domain.shape[0] == 2 or domain[2, 0] == domain[2, 1]:
            return np.array([x, y])
        else:
            z = random.uniform(*domain[2])
            return np.array([x, y, z])
    

"""
Creates coordinates for a point in the provided volume domain in two or three dimensions, 
defined by the weight factors which specify relative distances between min and max in the x, y, and z domains.
"""
def point_at_weights(domain: np.ndarray, weight_x: float, weight_y: float, weight_z: float=0):
    weights = weight_x, weight_y, weight_z
    return [(1-w)*d[0] + w*d[1] for w, d in zip(weights, domain)]


def weights_for_point(domain: np.ndarray, point):
    return [(position - dimension[0]) / (dimension[1] - dimension[0]) for dimension, position in zip(domain, point)]
