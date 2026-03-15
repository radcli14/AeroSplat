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

    def sample_evaluation_points(self):
        """Pre-sample boundary and volume points for one optimisation step.

        Returning a fixed set of points and reusing them for both the +δ and
        -δ loss evaluations (common random numbers) means the difference
        loss_p - loss_m reflects only the parameter perturbation, not
        independent sampling noise.  Without this the gradient estimate is
        dominated by noise whenever the per-point loss variance is comparable
        to the change in loss caused by the SPSA step.

        Returns
        -------
        boundary_points : list of lists
            boundary_points[i] is the list of sampled points for boundary i.
        volume_points : list of arrays
            Points sampled from the domain interior.
        """
        n_bnd = self.configuration["points_per_boundary"]
        n_vol = self.configuration["points_in_volume"]

        boundary_points = [
            [boundary.point_at_random() for _ in range(n_bnd)]
            for boundary in self.problem.boundaries
        ]

        # Use the current solution's sampler for a well-distributed interior sample.
        volume_points = self.solutions[-1].random_points(n_vol)

        return boundary_points, volume_points

    def boundary_loss(self, solution: AeroSplatSolution = None, boundary_points=None):
        solution = self.solutions[-1] if not solution else solution

        n_boundaries = len(self.problem.boundaries)
        n_points = self.configuration["points_per_boundary"]

        boundary_loss = 0.0
        for b_idx, boundary in enumerate(self.problem.boundaries):
            boundary_length = boundary.length
            points = boundary_points[b_idx] if boundary_points is not None \
                     else [boundary.point_at_random() for _ in range(n_points)]
            for point in points:
                velocity = solution.velocity_at(point)
                velocity_error = boundary.velocity - velocity
                # Use squared L2 norm as derived in the README: λ = e·e
                boundary_loss += boundary_length * np.linalg.norm(velocity_error)**2

        return boundary_loss / n_points / n_boundaries / self.problem.total_boundary_length / self.problem.velocity_scale**2

    def volume_loss(self, solution: AeroSplatSolution = None, volume_points=None):
        solution = self.solutions[-1] if not solution else solution

        n_points = self.configuration["points_in_volume"]
        points = volume_points if volume_points is not None \
                 else solution.random_points(n_points)

        volume_loss = 0.0
        for point in points:
            euler_terms = solution.euler_equation_terms_at(point)
            # Momentum residual: (v·∇)v components (all terms except last)
            # Normalised by v_scale^4 so it is dimensionally consistent with
            # the continuity term normalised by v_scale^2 below.
            momentum_residual = np.sum(euler_terms[:-1]**2) / self.problem.velocity_scale**2
            # Velocity-divergence (incompressibility): (∇·v)²
            continuity_residual = euler_terms[-1]**2
            # Mass-flux divergence: (∇·(ρv))² — enforces mass conservation when
            # density varies across the domain.
            mass_flux_residual = solution.mass_flux_divergence_at(point)**2

            volume_loss += momentum_residual + continuity_residual + mass_flux_residual

        return volume_loss / n_points / self.problem.velocity_scale**2

    def loss(self, solution: AeroSplatSolution = None,
             boundary_points=None, volume_points=None):
        boundary_loss = self.boundary_loss(solution, boundary_points)
        volume_loss = self.volume_loss(solution, volume_points)
        return boundary_loss + volume_loss

    def bernoulli_sequence(self):
        length_of_array = len(self.normalized_array())
        return 2.0 * np.random.binomial(1, 0.5, length_of_array) - 1

    def iterate(self):
        b = self.configuration["step_size_for_gradient_estimate"]
        c = self.configuration["step_size_for_update"]
        lam = self.configuration["gradient_weight"]

        theta = self.normalized_array()

        # Sample evaluation points once and reuse for both SPSA perturbations
        # (common random numbers).  This ensures loss_p - loss_m measures only
        # the effect of the parameter change, not independent sampling noise.
        boundary_points, volume_points = self.sample_evaluation_points()

        # Two-sided SPSA central-difference gradient estimate:
        #   ĝ = -(L(θ+δ) - L(θ-δ)) / (2δ),  bias O(δ²) vs O(δ) one-sided.
        delta_theta = b * self.bernoulli_sequence()

        solution_p = AeroSplatSolution.from_normalized_array(theta + delta_theta, self.problem.domain)
        solution_m = AeroSplatSolution.from_normalized_array(theta - delta_theta, self.problem.domain)
        loss_p = self.loss(solution_p, boundary_points, volume_points)
        loss_m = self.loss(solution_m, boundary_points, volume_points)

        # Central-difference gradient estimate; negate because we descend.
        gradient_estimate = -(loss_p - loss_m) / (2.0 * delta_theta)

        # Exponential moving average smooths noise without killing adaptability.
        self._gradient_estimate = lam * gradient_estimate + (1 - lam) * self._gradient_estimate

        # Create the next solution and record its loss with fresh sample points.
        theta_next = theta + c * self._gradient_estimate
        self.solutions.append(AeroSplatSolution.from_normalized_array(theta_next, self.problem.domain))
        self.history.append([self.boundary_loss(), self.volume_loss()])
