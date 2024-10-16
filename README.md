# pdet

This package defines ready-to-use neural network emulators for the LIGO-Virgo-KAGRA compact binary selection function.
Specifically, this package can be used to compute **detection probabilities** for compact binary parameters with a wide range of parameters.

The following table describes available emulators.

| Name | Observing run | Instruments | Valid Parameter Space |
| ---- | ------------- | ----------- | --------------------- |
| `pdet_O3` | O3 (includes both O3a and O3b) | LIGO-Hanford, LIGO-Livingston, Virgo | [pdet_O3](#pdet_o3)

## How to use

Once installed, detection probability calculations can be used in one of two ways, (1) [the `predict` method](#1-predict) and (2) directly calling the neural network via [the `__call__` method](#2-direct-evaluation-via-__call__)

### 1. `predict()`

The `predict` method allows for the evaluation of detection probabilities based on straightforward user-defined parameters.
An example is the following:

```python
from pdet import pdet_O3

# Create an emulator
p = pdet_O3()

# Define data
# As an example, consider parameters for three compact binaries
params = {
       'mass_1': [2.5, 20.0, 50.0],  # Primary source-frame mass (units Msun)
       'mass_2': [1.2, 20.0, 10.0],  # Secondary source-frame mass (units Msun)
       'a_1': [0.0, 0.2, 0.3],       # Primary dimensionless spin
       'a_2': [0.1, 0.4, 0.2],       # Secondary dimensionless spin
       'redshift': [0.1 ,0.4, 1.0]   # Redshift
}

# Compute detection probabilities
p.predict(params)
```

**Output**

```python
Array([[2.03875096e-14],
       [1.52368634e-02],
       [4.16787806e-14]], dtype=float64)
```

Compact binary parameters can be passed through any structure with key/value pairs, such as a dictionary as above, pandas DataFrame, or other structured array.

**Required parameters.** The following binary parameters are required:

* `mass_1`: Primary source-frame mass (units of solar masses)
* `mass_2`: Secondary source-frame mass (units of solar masses)
* `a_1`: Primary dimensionless spin
* `a_2`: Secondary dimensionless spin
* One of the following:
  * `redshift`: Redshift
  * `luminosity_distance`: Luminosity distance in Gpc
  * `comoving_distance`: Comoving distance in Gpc

**Optional parameters.**
The following parameters are, in contrast, optional.
Note that, if not provided, *they will be generated randomly for each binary* according to the default distributions listed below.

* `cos_theta_1`: Spin-orbit tilt of primary. If not provided, drawn uniformly between $`[-1,1]`$.
* `cos_theta_2`: Spin-orbit tilt of secondary. If not provided, drawn uniformly between $`[-1,1]`$.
* `phi_12`: Azimuthal angle between primary and secondary spin vectors (units of radians). If not provided, drawn uniformly between $`[0,2\pi]`$.
* `right_ascension`: Right ascension of binary on the sky (units of radians). If not provided, drawn uniformly between $`[0,2\pi]`$.
* `cos_inclination`: Cosine inclination of the binary's orbit with respect to our line of sight. If not provided, drawn uniformly between $`[-1,1]`$. The parameter `inclination` can equivalently be provided (units of radians)
* `polarization_angle`: Binary polarization angle. If not provided, drawn uniformly between $`[0,2\pi]`$.

### 2. Direct evaluation via `__call__()`

The `predict()` method above is not amenable to compilation and/or autodifferentiation in `jax`.
An alternative JIT-compileable and differentiable method is the `p.__call__()` function:

```python
from pdet import pdet_O3
import jax

# Instantiate trained emulator
p = pdet_O3()

# JIT compile
jitted_pdet_O3 = jax.jit(p)

# Define binary parameters.
m1 = [20., 30.]
m2 = [15., 29.]
a1 = [0.5, 0.9]
a2 = [0.3, 0.]
cost1 = [0.2, -0.7]
cost2 = [0.9, 0.]
z = [0.2, 0.5]
cos_inclination = [0.7, 1.]
pol = [0., 2.9]
phi12 = [1.2, 0.]
ra = [3.2, 0.5]
sin_dec = [-1.1, -0.7]
params = jnp.array([m1, m2, a1, a2, cost1, cost2, z, cos_inclination, pol, phi12, ra, sin_dec])

# Compute detection probabilities
jitted_pdet_O3(params)
```

**Output**

```python
Array([[0.55567697],
       [0.27248132]], dtype=float64)
```

## Range of validity

Networks were trained using data spanning the ranges described below:

### `pdet_O3`

Primary masses

* $1 M_\odot \leq m_1 \leq 100 M_\odot$

Secondary masses

* $m_2 \geq 1 M_\odot$ ($1 M_\odot \leq m_1 \leq 60 M_\odot$)
* $m_2 \geq 2 M_\odot$ ($60 M\_\odot \leq m\_1 \leq 100 M\_\odot$)
* Note: Although real pipeline injections were available only in the above intervals, auxiliary "hopeless" training data were generated with $m_2 \geq 1 M_\odot$ across the full range of primary masses, and so the network has learned *some* information outside these ranges.

Spin magnitudes

* $a<0.4$ (component masses below $2 M_\odot$)
* $a<0.998$ (component masses above $2 M_\odot$)
* Note: Although real pipeline injections were available only in the above intervals, auxiliary "hopeless" training data were generated with $a<0.998$ across the full range of component masses.
