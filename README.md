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

### Scaling
