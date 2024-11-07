import numpy as np
from scipy.optimize import fmin
from .problem import AeroSplatProblem
from .solution import AeroSplatSolution
from .functions import point_at_weights

class AeroSplatOptimizer:
    problem = None
    solutions = []

    configuration = {
        "points_per_boundary": 10,
        "points_in_volume": 50
    }

    def __init__(self, problem: AeroSplatProblem, initial: AeroSplatSolution, **kwargs):
        self.problem = problem
        self.solutions = [initial]
        for key, value in kwargs.items():
            self.configuration[key] = value
    
    def normalized_array(self, idx: int = -1):
        return self.solutions[idx].as_normalized_array

    def boundary_loss(self, idx: int = -1):
        solution = self.solutions[idx]

        n_boundaries = len(self.problem.boundaries)
        n_points = self.configuration["points_per_boundary"]

        boundary_loss = 0.0
        for boundary in self.problem.boundaries:
            for _ in range(n_points):
                point = boundary.point_at_random()
                velocity = solution.velocity_at(point)
                velocity_error = boundary.velocity - velocity
                boundary_loss += np.linalg.norm(velocity_error)
                
        return boundary_loss / n_points / n_boundaries / self.problem.velocity_scale

    def volume_loss(self, idx: int = -1):
        solution = self.solutions[idx]

        n_points = self.configuration["points_in_volume"]
        points = solution.random_points(n_points)

        #weights = solution.sampler.random(n=n_points)  # weights are from latin hypercube sampler
        compressibility_loss = 0.0
        for point in points:
            #point = point_at_weights(self.problem.domain, *weights[i, :])[:self.problem.ndims]
            euler_terms = solution.euler_equation_terms_at(point)
            compressibility_loss += euler_terms[-1]**2
            
        return compressibility_loss / self.problem.velocity_scale**2

    def loss(self, idx: int = -1):
        return self.boundary_loss(idx) + self.volume_loss(idx)
    