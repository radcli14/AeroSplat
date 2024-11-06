# Create Problem Definition Class
import numpy as np
from scipy.stats.qmc import LatinHypercube
import random
from .aerosplat import AeroSplat

class AeroSplatProblem:
    domain_x = np.zeros(2)
    domain_y = np.zeros(2)
    domain_z = np.zeros(2)
    
    splats = []
    boundaries = []
    
    def __init__(self, domain_x=[0, 0], domain_y=[0, 0], domain_z=[0, 0], boundaries=[], spawn: int = None):
        self.domain_x = np.array(domain_x)
        self.domain_y = np.array(domain_y)
        self.domain_z = np.array(domain_z)
        self.boundaries = boundaries

        # Initialize the latin hypercube sampler, which assures distributions are uniform on multiple dimensions
        self.sampler = LatinHypercube(d=self.ndims)

        # Determine to initialize with a spawned set of AeroSplat objects
        if spawn:
            self.spawn_random_splats(spawn)

    def __repr__(self):
        repr_str = "AeroSplatProblem("
        repr_str += f"\n  domain_x={self.domain_x}" if not all(self.domain_x == 0) else ""
        repr_str += f"\n  domain_y={self.domain_y}" if not all(self.domain_y == 0) else ""
        repr_str += f"\n  domain_z={self.domain_z}" if not all(self.domain_z == 0) else ""
        repr_str += f"\n  boundaries={self.boundaries}" if self.boundaries else ""
        repr_str += "\n)"
        return repr_str
    
    @property
    def ndims(self):
        return 3 if self.domain_z[1] != self.domain_z[0] else 2
    
    def point_at_random(self):
        x = random.uniform(*self.domain_x)
        y = random.uniform(*self.domain_y)
        z = random.uniform(*self.domain_z)
        return np.array([x, y, z]) if self.ndims == 3 else np.array([x, y])
    
    def point_at_weights(self, weight_x, weight_y, weight_z=0):
        weights = weight_x, weight_y, weight_z
        domains = self.domain_x, self.domain_y, self.domain_z
        return [(1-w)*d[0] + w*d[1] for w, d in zip(weights, domains)]

    def point_grid(self, nx=10, ny=10, nz=1):
        grid = [[[self.point_at_weights(k/(nx-1), j/(ny-1), i/(nz-1) if nz > 1 else 0) 
                  for k in range(nx)] for j in range(ny)] for i in range(nz)]
        grid = np.array(grid)
        # grid will be in [page->z][row->y][column->x] index ordering
        return grid if self.ndims == 3 else grid[0, :, :, :2]

    def random_splat(self, point=None):
        return AeroSplat(
            position=point if point else self.point_at_random(),
            velocity=np.array(np.random.normal(size=self.ndims)), #np.zeros(self.ndims),
            scale=10.0*np.array(np.random.uniform(size=self.ndims)), #0.1*np.ones(self.ndims),
            orientation=[np.pi * np.random.uniform(-1, 1)] if self.ndims == 2 else [1, 0, 0, 0]
            # TODO: random orientation in 3D mode
        )

    def spawn_random_splats(self, quantity, clear=True):
        self.splats = [] if clear else self.splats
        weights = self.sampler.random(n=quantity)  # weights are from latin hypercube sampler
        for i in range(quantity):
            point = self.point_at_weights(*weights[i, :])[:self.ndims]
            self.splats.append(self.random_splat(point))

    def velocity_at(self, point):
        return sum([splat.velocity_at(point) for splat in self.splats])

    def velocity_on_grid(self, nx=10, ny=10, nz=1):
        grid = self.point_grid(nx, ny, nz)
        velocity_grid = np.zeros(grid.shape)
        if self.ndims == 2:
            velocity_grid = [[self.velocity_at(grid[j, k]) for k in range(nx)] for j in range(ny)]
        else:
            velocity_grid = [[[self.velocity_at(grid[i, j, k]) for k in range(nx)] for j in range(ny)] for i in range(nz)]
        return np.array(velocity_grid)
    
    def velocity_gradient_at(self, point):
        return sum([splat.velocity_gradient_at(point) for splat in self.splats])

    def velocity_gradient_on_grid(self, nx=10, ny=10, nz=1):
        grid = self.point_grid(nx, ny, nz)
        gradient_grid = np.zeros(grid.shape)
        if self.ndims == 2:
            gradient_grid = [[self.velocity_gradient_at(grid[j, k]) for k in range(nx)] for j in range(ny)]
        else:
            gradient_grid = [[[self.velocity_gradient_at(grid[i, j, k]) for k in range(nx)] for j in range(ny)] for i in range(nz)]
        return np.array(gradient_grid)
    
    def velocity_gradient_matrix_at(self, point):
        return sum([splat.velocity_gradient_matrix_at(point) for splat in self.splats])

    def velocity_gradient_matrix_on_grid(self, nx=10, ny=10, nz=1):
        grid = self.point_grid(nx, ny, nz)
        gradient_matrix_grid = np.zeros(list(grid.shape) + [self.ndims])
        if self.ndims == 2:
            gradient_matrix_grid = [[self.velocity_gradient_matrix_at(grid[j, k]) for k in range(nx)] for j in range(ny)]
        else:
            gradient_matrix_grid = [[[self.velocity_gradient_matrix_at(grid[i, j, k]) for k in range(nx)] for j in range(ny)] for i in range(nz)]
        return np.array(gradient_matrix_grid)

    def euler_equation_terms_at(self, point):
        velocity = self.velocity_at(point)
        velocity_gradient_matrix = self.velocity_gradient_matrix_at(point)
        velocity_gradient = [velocity_gradient_matrix[i, i] for i in range(self.ndims)]
        return np.append(velocity @ velocity_gradient_matrix, np.sum(velocity_gradient))
    