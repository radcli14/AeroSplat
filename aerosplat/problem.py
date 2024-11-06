# Create Problem Definition Class
import numpy as np
import random

class AeroSplatProblem:
    domain_x = np.zeros(2)
    domain_y = np.zeros(2)
    domain_z = np.zeros(2)
    
    #splats = []
    boundaries = []
    
    def __init__(self, domain_x=[0, 0], domain_y=[0, 0], domain_z=[0, 0], boundaries=[]):
        self.domain_x = np.array(domain_x)
        self.domain_y = np.array(domain_y)
        self.domain_z = np.array(domain_z)
        self.boundaries = boundaries

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
