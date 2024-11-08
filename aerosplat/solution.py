# Create Solution Class
import numpy as np
from scipy.stats.qmc import LatinHypercube
import random
from .aerosplat import AeroSplat
from .functions import random_unit_quaternion, point_at_random, point_at_weights


class AeroSplatSolution:
    domain = None
    splats = []

    def __init__(self, domain: np.ndarray, spawn: int = None):
        self.domain = domain
        
        # Initialize the latin hypercube sampler, which assures distributions are uniform on multiple dimensions
        self.sampler = LatinHypercube(d=self.ndims)

        # Determine to initialize with a spawned set of AeroSplat objects
        if spawn:
            self.spawn_random_splats(spawn)

    @classmethod
    def from_normalized_array(cls, array: np.array, domain: np.ndarray):
        # Initialize the solution we will build from the provided array, initially without splats
        solution = cls(domain)
        solution.splats = []

        # Add the encoded splats to the solution
        ndims = 2 if len(domain) <= 2 or domain[2, 0] == domain[2, 1] else 3
        n_per_array = 8 if ndims == 2 else 14
        n_splats = int(len(array) / n_per_array)
        reshaped_array = array.reshape(n_splats, n_per_array)
        for theta in reshaped_array:
            solution.splats.append(AeroSplat.from_normalized_array(theta, domain))

        return solution

    @property
    def as_normalized_array(self):
        normalized_array = []
        for splat in self.splats:
            normalized_array += list(splat.as_normalized_array(self.domain))
        return np.array(normalized_array)

    @property
    def ndims(self):
        return 2 if self.domain[2, 0] == self.domain[2, 1] else 3

    def point_at_random(self):
        return point_at_random(self.domain)

    def point_at_weights(self, weight_x, weight_y, weight_z=0):
        return point_at_weights(self.domain, weight_x, weight_y, weight_z)

    def random_points(self, quantity: int=1):
        # weights are from latin hypercube sampler
        return [self.point_at_weights(*weights)[:self.ndims] for weights in self.sampler.random(n=quantity)]

    def random_splat(self, point=None):
        random_splat = AeroSplat.random_in(self.domain)
        random_splat.position = point
        return random_splat

    def spawn_random_splats(self, quantity, clear=True):
        self.splats = [] if clear else self.splats
        points = self.random_points(quantity)
        for point in points:
            self.splats.append(self.random_splat(point))

    def velocity_at(self, point):
        return sum([splat.velocity_at(point) for splat in self.splats])

    def velocity_on_grid(self, grid):
        velocity_grid = np.zeros(grid.shape)
        if self.ndims == 2:
            velocity_grid = [[self.velocity_at(point) for point in row] for row in grid]
        else:
            velocity_grid = [[[self.velocity_at(point) for point in row] for row in page] for page in grid]
        return np.array(velocity_grid)
    
    def velocity_gradient_at(self, point):
        return sum([splat.velocity_gradient_at(point) for splat in self.splats])

    def velocity_gradient_on_grid(self, grid):
        gradient_grid = np.zeros(grid.shape)
        if self.ndims == 2:
            gradient_grid = [[self.velocity_gradient_at(point) for point in row] for row in grid]
        else:
            gradient_grid = [[[self.velocity_gradient_at(point) for point in row] for row in page] for page in grid]
        return np.array(gradient_grid)
    
    def velocity_gradient_matrix_at(self, point):
        return sum([splat.velocity_gradient_matrix_at(point) for splat in self.splats])

    def velocity_gradient_matrix_on_grid(self, grid):
        gradient_matrix_grid = np.zeros(list(grid.shape) + [self.ndims])
        if self.ndims == 2:
            gradient_matrix_grid = [[self.velocity_gradient_matrix_at(point) for point in row] for row in grid]
        else:
            gradient_matrix_grid = [[[self.velocity_gradient_matrix_at(point) for point in row] for row in page] for page in grid]
        return np.array(gradient_matrix_grid)

    def euler_equation_terms_at(self, point):
        velocity = self.velocity_at(point)
        velocity_gradient_matrix = self.velocity_gradient_matrix_at(point)
        velocity_gradient = [velocity_gradient_matrix[i, i] for i in range(self.ndims)]
        return np.append(velocity @ velocity_gradient_matrix, np.sum(velocity_gradient))
    
    def euler_equation_terms_on_grid(self, grid):
        grid_shape = list(grid.shape)
        grid_shape[-1] += 1  # size of a single euler equation, if 2d, this has 3 dimensions, if 3d, it has 4
        euler_equation_terms_grid = np.zeros(grid_shape)
        if self.ndims == 2:
            euler_equation_terms_grid = [[self.euler_equation_terms_at(point) for point in row] for row in grid]
        else:
            euler_equation_terms_grid = [[[self.euler_equation_terms_at(point) for point in row] for row in page] for page in grid]
        return np.array(euler_equation_terms_grid)
