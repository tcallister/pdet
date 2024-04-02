# p-det-O3

This package defines ready-to-use neural network emulators for the LIGO-Virgo-KAGRA compact binary selection function.
Specifically, this package can be used to compute **detection probabilities** for compact binary parameters with a wide range of parameters.

The following table describes available emulators.

| Name | Observing run | Instruments | Valid Parameter Space |
| ---- | ------------- | ----------- | --------------------- |
| `p_det_O3` | O3 (includes both O3a and O3b) | LIGO | [test](#p_det_O3)

## How to use

Once installed, detection probability calculations can be used as follows:

```python
from pdet import p_det_O3

# Create an emulator
p = p_det_O3()

# Define data
# As an example, consider parameters for three compact binaries
params = {'mass_1':[2.5,10.0,15.0],  # Primary source-frame mass (units Msun)
          'mass_2':[1.2,5.0,10.0],   # Secondary source-frame mass (units Msun)
          'a_1':[0.0,0.2,0.3],       # Primary dimensionless spin
          'a_2':[0.1,0.4,0.2],       # Secondary dimensionless spin
          'redshift':[0.1,0.9,1.0]   # Redshift

# Compute detection probabilities
detection_probs = p.predict(params)
```

Compact binary parameters can be passed either via a dictionary, as in the above example, or via a pandas DataFrame.

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

## Range of validity

### `p_det_O3`
