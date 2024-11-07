from .problem import AeroSplatProblem
from .solution import AeroSplatSolution


class AeroSplatOptimizer:
    problem = None
    solutions = []

    def __init__(self, problem: AeroSplatProblem, initial_splat_count=4):
        self.problem = problem
        self.solutions.append(AeroSplatSolution(problem.domain, spawn=initial_splat_count))
    
    def boundary_loss(self, idx: int = -1):
        pass

    def volume_loss(self, idx: int = -1):
        pass

    def loss(self, idx: int = -1):
        pass