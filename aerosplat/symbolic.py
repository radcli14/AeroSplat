# Create Symbolic Functions
import numpy as np
from sympy import symbols, exp, lambdify, diff, Quaternion, Matrix

## Properties of the splat

x, y, z, u, v, w, sx, sy, sz, qw, qx, qy, qz, θ = symbols("x, y, z, u, v, w, s_x, s_y, s_z, q_w, q_x, q_y, q_z, θ")
position = np.array([x, y, z])
velocity = np.array([u, v, w])
scale = np.array([sx, sy, sz])
orientation_2d = θ
orientation_3d = np.array([qw, qx, qy, qz])
splat_properties_2d = [x, y, u, v, sx, sy, θ]
splat_properties_2d_example = [0, 0] + [1, 0] + [1, 1] + [0] + [2, 0]
splat_properties_3d = [x, y, z, u, v, w, sx, sy, sz, qw, qx, qy, qz]
splat_properties_3d_example = [0, 0, 0] + [1, 0, 0] + [1, 1, 1] + [1, 0, 0, 0] + [2, 0, 0] 


## Position for evaluation of the splat equations

xp, yp, zp = symbols("x_p, y_p, z_p")
eval_coords_2d = [xp, yp]
eval_coords_3d = [xp, yp, zp]
eval_at_position = np.array(eval_coords_3d)


## Equation for quaternion
"""
We provide two options for obtaining the quaternon.
In the case where a `numpy` array with four quaternion components are provided, we will obtain using the standard syntax `Quaternion(qw, qx, qy, qz)`.
In the case where a single component is provided, we will assume that this is a planar model with an angle evaluated about the $z$-axis, and obtain with the `Quaternion.from_axis_angle` class method.
We also will create the `idx_dimension` variable which shrinks subsequent matrices and vectors to use only the first two ($x$ and $y$) components.
"""

def quat(q):
    if isinstance(q, np.ndarray) :
        return Quaternion(*q) 
    else: 
        return Quaternion.from_axis_angle([0, 0, 1], q)
idx_dimension = lambda orientation: 3 if isinstance(orientation, np.ndarray) else 2

quaternion_2d = quat(orientation_2d)
quaternion_3d = quat(orientation_3d)


## Equation for gaussian
"""
The Gaussian equation that is created below uses the variance term _not-squared_, while in the plot I square this term.
There is reason for this; the variance that I will use below is going to be of the form

$\texttt{variance} = \sum_{i=1}^3 \left( \frac{x_i - \bar{x}_i}{s_i} \right)^2$

where $x_i$ is a position coordinate for where we are measuring, $\bar{x}_i$ is the position coordinate for the center of the Gaussian, and $s_i$ is a scale coefficient representing one standard deviation in the direction of that coordinate.
Traditionally, we would evaluate the Gaussian at some count of standard deviations, which would derive from the square-root of the above expression.
To avoid taking a square-root, just to subsequently square the term again, we create the Gaussian as a function of the variance directly.
However, the plot below shows an evaluation of the Gaussian with respect to standard deviations.
"""

gaussian = lambda variance: exp(-0.5 * variance)


## Equation for rotation matrix
"""
Here we create a function that obtains the rotation matrix from the quaternion, along with some simplification steps.
First, we substitute a value of 1 for the quaternion norm, as we will ensure that it is always a proper unit quaternion.
Second, we call the standard `simplify()` function.
"""

def rotation_matrix_eqn(orientation):
    quaternion = quat(orientation)
    rotation_matrix = quaternion.to_rotation_matrix().subs(quaternion.norm(), 1)
    rotation_matrix.simplify()
    idx = idx_dimension(orientation)
    return rotation_matrix[:idx, :idx]

rotation_eqn_2d = rotation_matrix_eqn(orientation_2d)
rotation_fcn_2d = lambdify(θ, rotation_eqn_2d)

rotation_eqn_3d = rotation_matrix_eqn(orientation_3d)
rotation_fcn_3d = lambdify(orientation_3d, rotation_eqn_3d)


## Equation for variance

def variance_eqn(orientation):
    rotation_matrix = rotation_matrix_eqn(orientation)
    idx = idx_dimension(orientation)
    dx = rotation_matrix[:idx, :idx] @ (eval_at_position[:idx] - position[:idx])
    s = dx / scale[:idx]
    return np.sum(s**2)

variance_eqn_2d = variance_eqn(orientation_2d)
variance_fcn_2d = lambdify(splat_properties_2d + eval_coords_2d, variance_eqn_2d)

variance_eqn_3d = variance_eqn(orientation_3d)
variance_fcn_3d = lambdify(splat_properties_3d + eval_coords_3d, variance_eqn_3d)


## Equation for variance derivatives

diff_variance_eqn_2d = [diff(variance_eqn_2d, variable) for variable in splat_properties_2d]
diff_variance_fcn_2d = lambdify(splat_properties_2d + eval_coords_2d, list(diff_variance_eqn_2d))

diff_variance_eqn_3d = [diff(variance_eqn_3d, variable) for variable in splat_properties_3d]
diff_variance_fcn_3d = lambdify(splat_properties_3d + eval_coords_3d, list(diff_variance_eqn_3d))

variance_gradient_eqn_2d = [diff(variance_eqn_2d, variable) for variable in eval_coords_2d]
variance_gradient_fcn_2d = lambdify(splat_properties_2d + eval_coords_2d, variance_gradient_eqn_2d)

variance_gradient_eqn_3d = [diff(variance_eqn_3d, variable) for variable in eval_coords_3d]
variance_gradient_fcn_3d = lambdify(splat_properties_3d + eval_coords_3d, variance_gradient_eqn_3d)


## Equation for gaussian derivatives

diff_gaussian_eqn_2d = [diff(gaussian(variance_eqn_2d), variable) for variable in splat_properties_2d]
diff_gaussian_fcn_2d = lambdify(splat_properties_2d + eval_coords_2d, list(diff_gaussian_eqn_2d))

diff_gaussian_eqn_3d = [diff(gaussian(variance_eqn_3d), variable) for variable in splat_properties_3d]
diff_gaussian_fcn_3d = lambdify(splat_properties_3d + eval_coords_3d, list(diff_gaussian_eqn_3d))

## Equation for velocity

def velocity_eqn(orientation):
    idx = idx_dimension(orientation)
    variance = variance_eqn(orientation)
    return velocity[:idx] * gaussian(variance)

velocity_eqn_2d = velocity_eqn(orientation_2d)
velocity_fcn_2d = lambdify(splat_properties_2d + eval_coords_2d, list(velocity_eqn_2d))

velocity_eqn_3d = velocity_eqn(orientation_3d)
velocity_fcn_3d = lambdify(splat_properties_3d + eval_coords_3d, list(velocity_eqn_3d))


## Equations for velocity derivatives
"""
Velocity is 1x2 or 1x3

Variance derivates are 1xN

$$
\frac{\partial \boldsymbol{v(\boldsymbol{x}, \boldsymbol{x}_p)} }{\partial x_i}
= 
\frac{\partial \boldsymbol{v_0(\boldsymbol{x})}}{\partial x_i}
g(\boldsymbol{x}, \boldsymbol{x}_p)
+
\boldsymbol{v_0(\boldsymbol{x})}
\frac{\partial g(\boldsymbol{x}, \boldsymbol{x}_p)}{\partial x_i}
$$
"""

velocity_gradient_eqn_2d = [diff(vk, pk) for vk, pk in zip(velocity_eqn_2d, eval_coords_2d)]
velocity_gradient_fcn_2d = lambdify(splat_properties_2d + eval_coords_2d, velocity_gradient_eqn_2d)

velocity_gradient_eqn_3d = [diff(vk, pk) for vk, pk in zip(velocity_eqn_3d, eval_coords_3d)]
velocity_gradient_fcn_3d = lambdify(splat_properties_3d + eval_coords_3d, velocity_gradient_eqn_3d)

def velocity_derivatives(ndim):
    if ndim == 2:
        return Matrix(np.concatenate([np.zeros([2, 2]), np.diag([1, 1]), np.zeros([3, 2])]))
    elif ndim == 3:
        return Matrix(np.concatenate([np.zeros([3, 3]), np.diag([1, 1, 1]), np.zeros([7, 3])]))
    
diff_velocity_eqn_2d = velocity_derivatives(2)*gaussian(variance_eqn_2d) + Matrix(diff_gaussian_eqn_2d) * Matrix([u, v]).transpose()
diff_velocity_fcn_2d = lambdify(splat_properties_2d + eval_coords_2d, diff_velocity_eqn_2d)

diff_velocity_eqn_3d = velocity_derivatives(3)*gaussian(variance_eqn_3d) + Matrix(diff_gaussian_eqn_3d) * Matrix([u, v, w]).transpose()
diff_velocity_fcn_3d = lambdify(splat_properties_3d + eval_coords_3d, diff_velocity_eqn_3d)