# Create Boundary Condition Class
import numpy as np
import random

class LineBoundary:
    point0 = np.zeros(3)
    point1 = np.zeros(3)
    velocity = np.zeros(3)
    
    def __init__(self, point0=[0, 0, 0], point1=[0, 0, 0], velocity=[0, 0, 0]):
        self.point0 = np.array(point0)
        self.point1 = np.array(point1)
        self.velocity = np.array(velocity)
        
    def __repr__(self):
        return f"LineBoundary(point0={self.point0}, point1={self.point1}, velocity={self.velocity})"
    
    @property
    def length(self):
        return np.linalg.norm(self.point1 - self.point0)

    def point_at(self, weighted_distance):
        return (1 - weighted_distance) * self.point0 + weighted_distance * self.point1
    
    def point_at_random(self):
        return self.point_at(random.uniform(0, 1))

    