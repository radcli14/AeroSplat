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
It is considered to be a function of the vector $\boldsymbol{r}_i$ representing the position of the $i^{th}$ splat relative to the origin, a second vector $\boldsymbol{r}_k$ for the $k^{th}$ point at which the splat is being evaluated, a vector $\boldsymbol{q}_i$ representing the orientation of the splat, and a vector $\boldsymbol{s}_i$ representing the scaling of the splat with respect to its principal axes.
We'll consider the combined vector $\boldsymbol{x}_i = \left( \boldsymbol{r}_i, \boldsymbol{q}_i, \boldsymbol{s}_i \right)$ to be the parameters of the splat, which we will use when designating derivatives or integrals taken with respect to the splat's own properties, versus those with respect to the point of evaluation.

### Distance

A distance vector 

```math
\boldsymbol{d}_{ik} = \boldsymbol{r}_k - \boldsymbol{r}_i
```

defines the vector from the center of the $i^{th}$ splat to the $k^{th}$ point of evaluation.
If we define

```math
\boldsymbol{r}_i = x_i \hat{\boldsymbol{i}} + y_i \hat{\boldsymbol{j}} + z_i \hat{\boldsymbol{k}}
```

and 

```math
\boldsymbol{r}_k = x_k  \hat{\boldsymbol{i}} + y_k \hat{\boldsymbol{j}} + z_k \hat{\boldsymbol{k}}
```

with $(\hat{\boldsymbol{i}}, \hat{\boldsymbol{j}}, \hat{\boldsymbol{k}})$ being the basis vectors for the inertial reference frame, then we can see

```math
\boldsymbol{d}_{ik} = (x_k - x_i) \hat{\boldsymbol{i}} + (y_k - y_i) \hat{\boldsymbol{j}} + (z_k - z_i) \hat{\boldsymbol{k}},
```

or, in matrix form,

```math
\boldsymbol{d} = \begin{bmatrix} x_k - x_i \\ y_k - y_i \\ z_k - z_i \end{bmatrix}.
```

This represents the distance in three-dimensional (3D) coordinates, however, we will initially develop our method for evaluation in a two-dimensional (2D) plane, for which we can eliminate the $z$ coordinate, in other words


```math
\boldsymbol{d}_{ik} = (x_k - x_i) \hat{\boldsymbol{i}} + (y_k - y_i) \hat{\boldsymbol{j}},
```

and 

```math
\boldsymbol{d}_{ik} = \begin{bmatrix} x_k - x_i \\ y_k - y_i \end{bmatrix}.
```

When representing the 2D variant of the splatting model in subsequent sections, we will only use the first two rows and/or columns of respective vectors and matrices.


### Orientation

Our splat is given an orientation with respect to the inertial reference frame, which we represent in the form of a rotation matrix $\mathbf{R}$.
For a model in 2D, where the orientation vector $\boldsymbol{q}_i$ is represented by a single angle $\theta$ in radians defined about $\hat{\boldsymbol{k}}$, the rotation matrix

```math
\mathbf{R}_i =
\left[\begin{matrix}\cos{\left(θ \right)} & - \sin{\left(θ \right)}\\\sin{\left(θ \right)} & \cos{\left(θ \right)}\end{matrix}\right].
```

For a model in 3D, where the orientation vector $\boldsymbol{q}= \left( q_w, q_x, q_y, q_z \right)$ contains components of a unit quaternion,

```math
\mathbf{R}_i = 
\left[\begin{matrix}- 2 q_{y}^{2} - 2 q_{z}^{2} + 1 & - 2 q_{w} q_{z} + 2 q_{x} q_{y} & 2 q_{w} q_{y} + 2 q_{x} q_{z}\\2 q_{w} q_{z} + 2 q_{x} q_{y} & - 2 q_{x}^{2} - 2 q_{z}^{2} + 1 & - 2 q_{w} q_{x} + 2 q_{y} q_{z}\\- 2 q_{w} q_{y} + 2 q_{x} q_{z} & 2 q_{w} q_{x} + 2 q_{y} q_{z} & - 2 q_{x}^{2} - 2 q_{y}^{2} + 1\end{matrix}\right].
```

Using either of these matrices, we may express the distance $\boldsymbol{d}_{ik}$ in the rotated frame as 

```math
\boldsymbol{c}_{ik} = \mathbf{R}_i \boldsymbol{d}_{ik},
```

using the $\boldsymbol{c}_{ik}$ to represent the distance vector in the rotated frame for the $k^{th}$ evaluation point relative to the $i^{th}$ splat.


### Scaling

We use the term scaling to represent stretching on shrinking a given splat along each of the principle axes in its rotated reference frame.
This scaled distance is obtained by taking the dot product of a scaling vector $\boldsymbol{s}_i = \left( s_x, s_y, s_z \right)$ times the rotated distance vector, or

```math
\boldsymbol{b}_{ik} = \boldsymbol{s}_i \cdot \mathbf{R}_i \boldsymbol{d}_{ik},
```

using the $\boldsymbol{b}$ to represent the _scaled_ distance vector in the rotated frame.
It is also convenient to write the above expression in matrix notation, or 

```math
\boldsymbol{b}_{ik} = \boldsymbol{s}_i^T \mathbf{R}_i\boldsymbol{d}_{ik},
```

where $(\cdot)^T$ is the matrix or vector transpose.
For example, in the 2D model, we can expand this as

```math
\boldsymbol{b}_{ik} =
\begin{bmatrix} s_x & s_y \end{bmatrix}
\left[\begin{matrix}\cos{\left(θ \right)} & - \sin{\left(θ \right)}\\\sin{\left(θ \right)} & \cos{\left(θ \right)}\end{matrix}\right]
\begin{bmatrix} x^\prime - x \\ y^\prime - y \end{bmatrix}.
```


### Variance

Given the equations above for how to position, rotate, and scale an individual splat, we can write the variance

```math
\sigma^2_{ik} = \boldsymbol{b}_{ik} \cdot \boldsymbol{b}_{ik},
```

and thus we may calculate the gaussian $g_{ik}$ as defined above given the splat properties $\boldsymbol{x}_i$ and evaluation location $\boldsymbol{r}_k$.

```
g_{ik} = e^{-0.5 \sigma^2_{ik}}
```

## Velocity

Having defined the properties of the Gaussian splat above, and assuming that the fluid associated with that splat has a velocity 

```math
\boldsymbol{v}_i = u_i \hat{\boldsymbol{i}} + v_i \hat{\boldsymbol{j}} + w_i \hat{\boldsymbol{k}}
```

at its center position $\boldsymbol{r}_i$, then the velocity of the splat fluid at the evaluation position $\boldsymbol{r}_k$ is

```math
\boldsymbol{v}_{ik} = \boldsymbol{v}_i g_{ik}.
```

Supposing we have several splats, where we will represent the quantity $n$, then we can write the total velocity of all fluid at that point as

```math
\boldsymbol{v}_k = \sum_{i=1}^n \boldsymbol{v}_{i} g_{ik}.
```

### Boundary Error

We will define a boundary condition where the velocity is specified as

```math
\boldsymbol{v}_b = u_b \hat{\boldsymbol{i}} + v_b \hat{\boldsymbol{j}} + w_b \hat{\boldsymbol{k}}
```

using the $b$ subscript to represent an individual boundary, which may exist on either a line or surface.
In general, this will be some fluid source, for which one or more of the terms is non-zero, or a non-slip boundary on an object's surface where are of the terms are zero.
A velocity error vector $\boldsymbol{e}_{bk}$ is defined on the $b^{th}$ boundary for the $k^{th}$ evaluation point,

```math
\boldsymbol{e}_{bk} = \boldsymbol{v}_k - \boldsymbol{v}_b,
```

and there is an associated error variance

```math
\lambda_{bk} = \boldsymbol{e}_{bk} \cdot \boldsymbol{e}_{bk}.
```

As part of defining our loss function for optimizing the properties of our splats, we would intend to integrate our error variance over the boundary line or surface, so our loss at the $b^{th}$ boundary is

```math
L_b = \int_b \lambda_{bk} \textnormal{d} b .
```

In the case of a line boundary ...

In the case of a surface boundary ...

### Volume Error

