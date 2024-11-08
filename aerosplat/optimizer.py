import numpy as np
from scipy.optimize import fmin
from .problem import AeroSplatProblem
from .solution import AeroSplatSolution

class AeroSplatOptimizer:
    problem = None
    solutions = []
    history = []

    configuration = {
        "points_per_boundary": 10,
        "points_in_volume": 50,
        "step_size_for_gradient_estimate": 0.1,
        "step_size_for_update": 0.2,
        "gradient_weight": 0.2
    }

    _gradient_estimate = None

    def __init__(self, problem: AeroSplatProblem, initial: AeroSplatSolution, **kwargs):
        self.problem = problem
        self.solutions = [initial]
        for key, value in kwargs.items():
            self.configuration[key] = value

        # Initial gradient estimate for the splat parameters is zero
        self._gradient_estimate = 0.0 * initial.as_normalized_array
        
        # Initial loss function
        self.history = []
        self.history.append([self.boundary_loss(), self.volume_loss()])
    
    def normalized_array(self, solution: AeroSplatSolution = None):
        solution = self.solutions[-1] if not solution else solution
        return solution.as_normalized_array

    def boundary_loss(self, solution: AeroSplatSolution = None):
        solution = self.solutions[-1] if not solution else solution

        n_boundaries = len(self.problem.boundaries)
        n_points = self.configuration["points_per_boundary"]

        boundary_loss = 0.0
        for boundary in self.problem.boundaries:
            boundary_length = boundary.length
            for _ in range(n_points):
                point = boundary.point_at_random()
                velocity = solution.velocity_at(point)
                velocity_error = boundary.velocity - velocity
                boundary_loss += boundary_length * np.linalg.norm(velocity_error)
                
        return boundary_loss / n_points / n_boundaries / self.problem.total_boundary_length / self.problem.velocity_scale

    def volume_loss(self, solution: AeroSplatSolution = None):
        solution = self.solutions[-1] if not solution else solution

        n_points = self.configuration["points_in_volume"]
        points = solution.random_points(n_points)

        compressibility_loss = 0.0
        for point in points:
            euler_terms = solution.euler_equation_terms_at(point)
            compressibility_loss += euler_terms[-1]**2
            
        return compressibility_loss / n_points / self.problem.velocity_scale**2

    def loss(self, solution: AeroSplatSolution = None):
        boundary_loss = self.boundary_loss(solution)
        volume_loss = self.volume_loss(solution)
        return boundary_loss + volume_loss
    
    def bernoulli_sequence(self):
        length_of_array = len(self.normalized_array())
        return 2.0 * np.random.binomial(1, 0.5, length_of_array) - 1
    
    def iterate(self):
        b = self.configuration["step_size_for_gradient_estimate"]
        c = self.configuration["step_size_for_update"]
        lam = self.configuration["gradient_weight"]

        # Prior result
        theta = self.normalized_array()
        theta_loss = sum(self.history[-1])

        # Solution to estimate a gradient
        delta_theta_p = b * self.bernoulli_sequence()
        theta_p = theta + delta_theta_p
        solution_p = AeroSplatSolution.from_normalized_array(theta_p, self.problem.domain)
        solution_p_loss = self.loss(solution_p)
        delta_loss = solution_p_loss - theta_loss
        
        # Update to gradient estimate
        self._gradient_estimate = lam * (-delta_loss / delta_theta_p) + (1 - lam) * self._gradient_estimate

        # Create the next solution
        theta_next = theta + c * self._gradient_estimate
        self.solutions.append(AeroSplatSolution.from_normalized_array(theta_next, self.problem.domain))
        self.history.append([self.boundary_loss(), self.volume_loss()])
