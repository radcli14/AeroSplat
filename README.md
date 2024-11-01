# AeroSplat
Experimenting with splatting methods for aerodynamics analysis

## Gaussian Curve

We define our Guassian curve as a function of variance, which we will represent by the symbol $\sigma^2$.
In this case, $\sigma$ will be referred to as a deviation, however, for our application we will calculate $\sigma^2$ directly.
Evaluated at a single point, this is given by 

$g = e^{-0.5 \sigma^2}$,

and evaluating for a range in $\sigma$ yields the curve below.

![Gaussian evaluations with respect to $\sigma$](images/gaussian.png)


## Gaussian Splat

The Gaussian splat will be used to scale the velocity given at the center of the splat for evaluations at some distance away from its center.
It is considered to be a function of the vector $\boldsymbol{r}$ representing the position of the splat relative to the origin, a second vector $\boldsymbol{r}^\prime$ for the point at which the splat is being evaluated, a vector $\boldsymbol{q}$ representing the orientation of the splat, and a vector $\boldsymbol{s}$ representing the scaling of the splat with respect to its principal axes.
We'll consider the combined vector $\boldsymbol{x} = \left( \boldsymbol{r}, \boldsymbol{q}, \boldsymbol{s} \right)$ to be the parameters of the splat, which we will use when designating derivatives or integrals taken with respect to the splat's own properties, versus those will respect to the point of evaluation.

### Distance

A distance vector 

```math
\boldsymbol{d} = \boldsymbol{r}^\prime - \boldsymbol{r}
```

defines the vector from the center of the splat to the point of evaluation.
If we define

```math
\boldsymbol{r} = x \hat{\boldsymbol{i}} + y \hat{\boldsymbol{j}} + z \hat{\boldsymbol{k}}
```

and 

```math
\boldsymbol{r}^\prime = x^\prime  \hat{\boldsymbol{i}} + y^\prime  \hat{\boldsymbol{j}} + z^\prime  \hat{\boldsymbol{k}}
```

with $(\hat{\boldsymbol{i}}, \hat{\boldsymbol{j}}, \hat{\boldsymbol{k}})$ being the basis vectors for the inertial reference frame, then we can see

```math
\boldsymbol{d} = (x^\prime - x) \hat{\boldsymbol{i}} + (y^\prime - y) \hat{\boldsymbol{j}} + (z^\prime - z) \hat{\boldsymbol{k}},
```

or, in matrix form,

```math
\boldsymbol{d} = \begin{bmatrix} x^\prime - x \\ y^\prime - y \\ z^\prime - z \end{bmatrix}.
```

This represents the distance in three-dimensional (3D) coordinates, however, we will initially develop our method for evaluation in a two-dimensional (2D) plane, for which we can eliminate the $z$ coordinate, in other words


```math
\boldsymbol{d} = (x^\prime - x) \hat{\boldsymbol{i}} + (y^\prime - y) \hat{\boldsymbol{j}},
```

and 

```math
\boldsymbol{d} = \begin{bmatrix} x^\prime - x \\ y^\prime - y \end{bmatrix}.
```

When representing the 2D variant of the splatting model in subsequent sections, we will only use the first two rows and/or columns of respective vectors and matrices.


### Orientation

Our splat is given an orientation with respect to the inertial reference frame, which we represent in the form of a rotation matrix $\mathbf{R}$.
For a model in 2D, where the orientaition vector $\boldsymbol{q}$ is represented by a single angle $\theta$ in radians defined about $\hat{\boldsymbol{k}}$, the rotation matrix

```math
\mathbf{R} =
\left[\begin{matrix}\cos{\left(θ \right)} & - \sin{\left(θ \right)}\\\sin{\left(θ \right)} & \cos{\left(θ \right)}\end{matrix}\right].
```

For a model in 3D, where the orientation vector $\boldsymbol{q}= \left( q_w, q_x, q_y, q_z \right)$ contains components of a unit quaternion,

```math
\mathbf{R} = 
\left[\begin{matrix}- 2 q_{y}^{2} - 2 q_{z}^{2} + 1 & - 2 q_{w} q_{z} + 2 q_{x} q_{y} & 2 q_{w} q_{y} + 2 q_{x} q_{z}\\2 q_{w} q_{z} + 2 q_{x} q_{y} & - 2 q_{x}^{2} - 2 q_{z}^{2} + 1 & - 2 q_{w} q_{x} + 2 q_{y} q_{z}\\- 2 q_{w} q_{y} + 2 q_{x} q_{z} & 2 q_{w} q_{x} + 2 q_{y} q_{z} & - 2 q_{x}^{2} - 2 q_{y}^{2} + 1\end{matrix}\right].
```

Using either of these matrices, we may express the $\boldsymbol{d}$ in the rotated frame as 

```math
\boldsymbol{d}_r = \mathbf{R}\boldsymbol{d},
```

using the subscripted $\boldsymbol{d}_r$ to represent the distance vector in the rotated frame.


### Scaling

We use the term scaling to represent stretching on shrinking a given splat along each of the principle axes in its rotated reference frame.
This scaled distance is obtained by taking the dot product of a scaling vector $\boldsymbol{s} = \left( s_x, s_y, s_z \right)$ times the rotated distance vector, or

```math
\boldsymbol{d}_s = \boldsymbol{s} \cdot \mathbf{R} \boldsymbol{d},
```

using the subscripted $\boldsymbol{d}_s$ to represent the _scaled_ distance vector in the rotated frame.
It is also convenient to write the above expression in matrix notation, or 

```math
\boldsymbol{d}_s = \boldsymbol{s}^T \mathbf{R}\boldsymbol{d},
```

where $(\cdot)^T$ is the matrix or vector transpose.
For example, in the 2D model, we can expand this as

```math
\boldsymbol{d}_s =
\begin{bmatrix} s_x & s_y \end{bmatrix}
\left[\begin{matrix}\cos{\left(θ \right)} & - \sin{\left(θ \right)}\\\sin{\left(θ \right)} & \cos{\left(θ \right)}\end{matrix}\right]
\begin{bmatrix} x^\prime - x \\ y^\prime - y \end{bmatrix}.
```


### Variance

Given the equations above for how to position, rotate, and scale an individual splat, we can write the variance

```math
\sigma^2 = \boldsymbol{d}_s \cdot \boldsymbol{d}_s,
```

and thus we may calculate the gaussian $g$ as defined above given the splat properties $\boldsymbol{x}$ and evaluation location $\boldsymbol{r}^\prime$.


## Velocity

Having defined the properties of the Gaussian splat above, and assuming that the fluid associated with that splat has a velocity 

```math
\boldsymbol{v} = u \hat{\boldsymbol{i}} + v \hat{\boldsymbol{j}} + w \hat{\boldsymbol{k}}
```

defined at its center position $\boldsymbol{r}$, then the velocity of the splat fluid at the evaluation position $\boldsymbol{r}^\prime$ is

```math
\boldsymbol{v}^\prime = \boldsymbol{v} g.
```

Supposing we have several splates, where we will represent the quantity $n$, and we represent a single splat using the subscript $i$, then we can write the total velocity of all fluid at that point as

```math
\boldsymbol{v}^\prime = \sum_{i=1}^n \boldsymbol{v}_i g.
```
