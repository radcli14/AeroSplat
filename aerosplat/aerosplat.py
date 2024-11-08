# Create Splatting Class
import numpy as np
from .symbolic import *
from .functions import point_at_random, point_at_weights, weights_for_point, normalize_quaternion, random_unit_quaternion, length_scale


class AeroSplat:
    position = np.zeros(3)
    velocity_vector = np.array([1, 0, 0])
    velocity_magnitude = 0
    scale = np.ones(3)
    orientation = np.array([1, 0, 0, 0])
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, np.array(value))
        
        # Make sure the velocity vector is a unit vector
        self.velocity_vector = self.velocity_vector / np.linalg.norm(self.velocity_vector) 

        # Make sure the orientation has valid dimension
        if not self.orientation.shape:
            self.orientation = np.array([self.orientation])  # bumps from zero order to first order
    
    def __repr__(self):
        return f"AeroSplat(position={self.position}, velocity={self.velocity}, scale={self.scale}, orientation={self.orientation})"
    
    @classmethod
    def random_in(cls, domain: np.ndarray, velocity_scale: float=1.0):
        ndims = 2 if len(domain) == 2 or domain[2, 0] == domain[2, 1] else 3
        return cls(
            position = point_at_random(domain),
            velocity_vector = np.random.normal(size=ndims),
            velocity_magnitude = velocity_scale * np.exp(np.random.normal()),
            scale = 10 / length_scale(domain) * np.exp(np.random.normal(size=ndims)),
            orientation = [np.pi * np.random.uniform(-1, 1)] if ndims == 2 else random_unit_quaternion()
        )

    @classmethod
    def from_normalized_array(cls, theta: np.array, domain: np.ndarray):
        ndims = 2 if len(theta) == 8 else 3
        return cls(
            position = point_at_weights(domain, *theta[:ndims])[:ndims],
            velocity_vector = theta[ndims:2*ndims],
            velocity_magnitude = np.exp(theta[2*ndims]),
            scale = np.exp(theta[2*ndims+1:3*ndims+1]),
            orientation = theta[3*ndims+1] if ndims == 2 else normalize_quaternion(theta[3*ndims+1:])
        )

    def as_normalized_array(self, domain: np.ndarray):
        normalized_array = []
        normalized_array += list(weights_for_point(domain, self.position))
        normalized_array += list(self.velocity_vector)
        normalized_array.append(np.log(self.velocity_magnitude))
        normalized_array += list(np.log(self.scale))
        normalized_array += list(self.orientation)
        return normalized_array

    @property
    def is2d(self):
        return len(self.position) == 2
    
    @property
    def velocity(self):
        return self.velocity_magnitude * self.velocity_vector
        #TODO: create a setter for the velocity property. I stored the velocity_magnitude and velocity_vector separately to aid in log-normalization of the magnitude

    @property
    def properties(self):
        return np.concatenate([self.position, self.velocity, self.scale, self.orientation])
    
    @property
    def quaternion(self):
        return quat(*self.orientation) if self.is2d else quat(self.orientation)
    
    @property
    def rotation_matrix(self):
        return rotation_fcn_2d(*self.orientation) if self.is2d else rotation_fcn_3d(*self.orientation)
    
    @property
    def scale_matrix(self):
        return np.diag(self.scale)
    
    def properties_at(self, position):
        return list(self.properties) + list(position)

    def variance_at(self, position):
        f = variance_fcn_2d if self.is2d else variance_fcn_3d
        return f(*self.properties_at(position))

    def variance_gradient_at(self, position):
        f = variance_gradient_fcn_2d if self.is2d else variance_gradient_fcn_3d
        gradient = f(*self.properties_at(position))
        return np.array(gradient)

    def gaussian_at(self, position):
        return np.exp(-0.5 * self.variance_at(position))

    def gaussian_gradient_at(self, position):
        return -0.5 * self.variance_gradient_at(position) * self.gaussian_at(position)

    def velocity_at(self, position):
        return self.velocity * self.gaussian_at(position)
    
    def velocity_gradient_matrix_at(self, position):
        f = velocity_gradient_matrix_fcn_2d if self.is2d else velocity_gradient_matrix_fcn_3d
        gradient_matrix = f(*self.properties_at(position))
        return np.array(gradient_matrix)

    def velocity_gradient_at(self, position):
        return self.velocity * self.gaussian_gradient_at(position)
    
    def differential_velocity_at(self, position):
        f = diff_velocity_fcn_2d if self.is2d else diff_velocity_fcn_3d
        return f(*self.properties_at(position))
