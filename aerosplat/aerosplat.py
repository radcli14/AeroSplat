# Create Splatting Class
from .symbolic import *

class AeroSplat:
    position = np.zeros(3)
    velocity = np.zeros(3)
    scale = np.ones(3)
    orientation = np.array([1, 0, 0, 0])
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, np.array(value))
    
    def __repr__(self):
        return f"AeroSplat(position={self.position}, velocity={self.velocity}, scale={self.scale}, orientation={self.orientation})"
    
    @property
    def is2d(self):
        return len(self.position) == 2
    
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
    
    def velocity_gradient_at(self, position):
        return self.velocity * self.gaussian_gradient_at(position)

    def differential_velocity_at(self, position):
        f = diff_velocity_fcn_2d if self.is2d else diff_velocity_fcn_3d
        return f(*self.properties_at(position))
