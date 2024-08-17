import json
import os
import warnings

import equinox as eqx
import h5py
import jax
import numpy as np
import pandas
from astropy import units as u
from astropy.cosmology import Planck15, z_at_value
from jax import numpy as jnp


jax.config.update("jax_enable_x64", True)


class emulator:
    """
    Base class implementing a generic detection probability emulator.
    Intended to be subclassed when constructing emulators for particular
    networks/observing runs.
    """

    def __init__(
        self,
        trained_weights,
        scaler,
        input_size,
        hidden_layer_width,
        hidden_layer_depth,
        activation,
        final_activation,
    ):
        """
        Instantiate an `emulator` object

        Parameters
        ----------
        trained_weights : `str`
            Filepath to .hdf5 file containing trained network weights, as
            saved by a `tensorflow.keras.Model.save_weights` command
        scaler : `str`
            Filepath to saved `sklearn.preprocessing.StandardScaler` object,
            fitted during network training
        input_size : `int`
            Dimensionality of input feature vector
        hidden_layer_width : `int`
            Width of hidden layers
        hidden_layer_depth : `int`
            Number of hidden layers
        activation : `func`
            Activation function to be applied to hidden layers

        Returns
        -------
        None
        """

        # Instantiate neural network
        self.trained_weights = trained_weights
        self.nn = eqx.nn.MLP(
            in_size=input_size,
            out_size=1,
            depth=hidden_layer_depth,
            width_size=hidden_layer_width,
            activation=activation,
            final_activation=final_activation,
            key=jax.random.PRNGKey(111),
        )

        # Load trained weights and biases
        weight_data = h5py.File(self.trained_weights, "r")

        # Load scaling parameters
        with open(scaler, "r") as f:
            self.scaler = json.load(f)
            self.scaler["mean"] = jnp.array(self.scaler["mean"])
            self.scaler["scale"] = jnp.array(self.scaler["scale"])

        # Define helper functions with which to access MLP weights and biases
        # Needed by `eqx.tree_at`
        def get_weights(i):
            return lambda t: t.layers[i].weight

        def get_biases(i):
            return lambda t: t.layers[i].bias

        # Loop across layers, load pre-trained weights and biases
        for i in range(hidden_layer_depth + 1):
            if i == 0:
                key = "dense"
            else:
                key = "dense_{0}".format(i)

            layer_weights = weight_data["{0}/{0}/kernel:0".format(key)][()].T
            self.nn = eqx.tree_at(get_weights(i), self.nn, layer_weights)

            layer_biases = weight_data["{0}/{0}/bias:0".format(key)][()].T
            self.nn = eqx.tree_at(get_biases(i), self.nn, layer_biases)

    def _transform_parameters(self, *physical_params):
        """
        OVERWRITE UPON SUBCLASSING.

        Function to convert from a predetermined set of user-provided physical
        CBC parameters to the input space expected by the trained neural
        network. Used by `emulator.__call__` below.

        NOTE: This function should be JIT-able and differentiable, and so
        consistency/completeness checks should be performed upstream; we
        should be able to assume that `physical_params` is provided as
        expected.

        Parameters
        ----------
        physical_params : numpy.array or jax.numpy.array
            Array containing physical parameters characterizing CBC signals

        Returns
        -------
        transformed_parameters : jax.numpy.array
            Transformed parameter space expected by trained neural network
        """

        # APPLY REQUIRED TRANSFORMATION HERE
        # transformed_params = ...

        # Dummy transformation
        transformed_params = physical_params

        # Jaxify
        transformed_params = jnp.array(physical_params)

        return transformed_params

    def __call__(self, x):
        """
        Function to evaluate the trained neural network on a set of user-
        provided physical CBC parameters.

        NOTE: This function should be JIT-able and differentiable, and so any
        consistency or completeness checks should be performed upstream, such
        that we can assume the provided parameter vector `x` is already in the
        correct format expected by the `emulator._transform_parameters` method.
        """

        # Transform physical parameters to space expected by the neural network
        transformed_x = self._transform_parameters(*x)

        # Apply scaling, evaluate the network, and return
        scaled_x = (transformed_x.T - self.scaler["mean"]) / self.scaler["scale"]
        return jax.vmap(self.nn)(scaled_x)

    def _check_distance(self, parameter_dict):
        """
        Helper function to check the presence of required distance arguments,
        and augment input parameters with additional quantities as needed.

        Parameters
        ----------
        parameter_dict : `dict` or `pandas.DataFrame`
            Set of compact binary parameters for which we want to evaluate Pdet

        Returns
        -------
        None
        """

        # Check for distance parameters
        # If none are present, or if more than one is present, return an error
        allowed_distance_params = [
            "luminosity_distance",
            "comoving_distance",
            "redshift",
        ]
        if not any(param in parameter_dict for param in allowed_distance_params):
            raise RuntimeError(
                "Missing distance parameter. Requires one of:", allowed_distance_params
            )
        elif sum(param in parameter_dict for param in allowed_distance_params) > 1:
            raise RuntimeError(
                "Multiple distance parameters present. Only one of the following allowed:",
                allowed_distance_params,
            )

        # Augment, such both redshift and luminosity distance are present
        if "comoving_distance" in parameter_dict:
            parameter_dict["redshift"] = z_at_value(
                Planck15.comoving_distance, parameter_dict["comoving_distance"] * u.Gpc
            ).value
            parameter_dict["luminosity_distance"] = (
                Planck15.luminosity_distance(parameter_dict["redshift"]).to(u.Gpc).value
            )

        elif "luminosity_distance" in parameter_dict:
            parameter_dict["redshift"] = z_at_value(
                Planck15.luminosity_distance,
                parameter_dict["luminosity_distance"] * u.Gpc,
            ).value

        elif "redshift" in parameter_dict:
            parameter_dict["luminosity_distance"] = (
                Planck15.luminosity_distance(parameter_dict["redshift"]).to(u.Gpc).value
            )

    def _check_masses(self, parameter_dict):
        """
        Helper function to check the presence of required mass arguments, and
        augment input parameters with additional quantities needed for
        prediction.

        Parameters
        ----------
        parameter_dict : `dict` or `pandas.DataFrame`
            Set of compact binary parameters for which we want to evaluate Pdet

        Returns
        -------
        None
        """

        # Check that mass parameters are present
        required_mass_params = ["mass_1", "mass_2"]
        for param in required_mass_params:
            if param not in parameter_dict:
                raise RuntimeError("Must include {0} parameter".format(param))

        # Reshape for safety below
        parameter_dict["mass_1"] = np.reshape(parameter_dict["mass_1"], -1)
        parameter_dict["mass_2"] = np.reshape(parameter_dict["mass_2"], -1)

    def _check_spins(self, parameter_dict):
        """
        Helper function to check for the presence of required spin parameters
        and augment with additional quantities as needed.

        Parameters
        ----------
        parameter_dict : `dict` or `pandas.DataFrame`
            Set of compact binary parameters for which we want to evaluate Pdet

        Returns
        -------
        None
        """

        # Check that required spin parameters are present
        required_spin_params = ["a_1", "a_2"]
        for param in required_spin_params:
            if param not in parameter_dict:
                raise RuntimeError("Must include {0} parameter".format(param))

        # Reshape for safety below
        parameter_dict["a_1"] = np.reshape(parameter_dict["a_1"], -1)
        parameter_dict["a_2"] = np.reshape(parameter_dict["a_2"], -1)

        # Check for optional parameters, fill in if absent
        if "cos_theta_1" not in parameter_dict:
            warnings.warn(
                "Parameter 'cos_theta_1' not present. Filling with random value from isotropic distribution."
            )
            parameter_dict["cos_theta_1"] = (
                2.0 * np.random.random(parameter_dict["a_1"].shape) - 1.0
            )
        if "cos_theta_2" not in parameter_dict:
            warnings.warn(
                "Parameter 'cos_theta_2' not present. Filling with random value from isotropic distribution."
            )
            parameter_dict["cos_theta_2"] = (
                2.0 * np.random.random(parameter_dict["a_2"].shape) - 1.0
            )
        if "phi_12" not in parameter_dict:
            warnings.warn(
                "Parameter 'phi_12' not present. Filling with random value from isotropic distribution."
            )
            parameter_dict["phi_12"] = (
                2.0 * np.pi * np.random.random(parameter_dict["a_1"].shape)
            )

    def _check_extrinsic(self, parameter_dict):
        """
        Helper method to check required extrinsic parameters and augment as
        necessary.

        Parameters
        ----------
        parameter_dict : `dict` or `pandas.DataFrame`
            Set of compact binary parameters for which we want to evaluate Pdet

        Returns
        -------
        None
        """

        if "right_ascension" not in parameter_dict:
            warnings.warn(
                "Parameter 'right_ascension' not present. Filling with random value from isotropic distribution."
            )
            parameter_dict["right_ascension"] = (
                2.0 * np.pi * np.random.random(parameter_dict["mass_1"].shape)
            )

        if "sin_declination" not in parameter_dict:
            warnings.warn(
                "Parameter 'sin_declination' not present. Filling with random value from isotropic distribution."
            )
            parameter_dict["sin_declination"] = (
                2.0 * np.random.random(parameter_dict["mass_1"].shape) - 1.0
            )

        if "inclination" not in parameter_dict:
            if "cos_inclination" not in parameter_dict:
                warnings.warn(
                    "Parameter 'inclination' or 'cos_inclination' not present. Filling with random value from isotropic distribution."
                )
                parameter_dict["cos_inclination"] = (
                    2.0 * np.random.random(parameter_dict["mass_1"].shape) - 1.0
                )
        else:
            parameter_dict["cos_inclination"] = np.cos(parameter_dict["inclination"])

        if "polarization_angle" not in parameter_dict:
            warnings.warn(
                "Parameter 'polarization_angle' not present. Filling with random value from isotropic distribution."
            )
            parameter_dict["polarization_angle"] = np.pi * np.random.random(
                parameter_dict["mass_1"].shape
            )

    def check_input(self, parameter_dict):
        """
        Method to check provided set of compact binary parameters for any
        missing information, and/or to augment provided parameters with any
        additional derived information expected by the neural network. If
        extrinsic parameters (e.g. sky location, polarization angle, etc.) have
        not been provided, they will be randomly generated and appended to the
        given CBC parameters.

        Parameters
        ----------
        parameter_dict : `dict` or `pandas.DataFrame`
            Set of compact binary parameters for which we want to evaluate Pdet

        Returns
        -------
        parameter_dict : `dict`
            Dictionary of CBC parameters, augmented with necessary derived
            parameters
        """

        # Convert from pandas table to dictionary, if necessary
        if type(parameter_dict) == pandas.core.frame.DataFrame:
            parameter_dict = parameter_dict.to_dict(orient="list")

        # Check parameters
        self._check_distance(parameter_dict)
        self._check_masses(parameter_dict)
        self._check_spins(parameter_dict)
        self._check_extrinsic(parameter_dict)

        return parameter_dict


class pdet_O3(emulator):
    """
    Class implementing the LIGO-Hanford, LIGO-Livingston, and Virgo network's
    selection function during their O3 observing run. Used to evaluate the detection
    probability of compact binaries, assuming a false alarm threshold of below
    1 per year. The computed detection probabilities include all variation in
    the detectors' sensitivities over the course of the O3 run and accounts for
    time in which the instruments were not in observing mode. They should
    therefore be interpreted as the probability of a CBC detection if that CBC
    occurred during a random time between the startdate and enddate of O3.
    """

    def __init__(self, model_weights=None, scaler=None):
        """
        Instantiates a `p_det_O3` object, subclassed from the `emulator` class.

        Parameters
        ----------
        model_weights : `None` or `str`
            Filepath to .hdf5 file containing trained network weights, as saved
            by a `tensorflow.keras.Model.save_weights`, command, if one wishes
            to override the provided default weights (which are loaded when
            `model_weights==None`).
        scaler : `str`
            Filepath to saved `sklearn.preprocessing.StandardScaler` object, if
            one wishes to override the provided default (loaded when
            `scaler==None`).
        """

        if model_weights is None:
            model_weights = os.path.join(
                os.path.dirname(__file__), "./../trained_weights/weights_HLV_O3.hdf5"
            )
        else:
            print("Overriding default weights")

        if scaler is None:
            scaler = os.path.join(
                os.path.dirname(__file__), "./../trained_weights/scaler_HLV_O3.json"
            )
        else:
            print("Overriding default weights")

        input_dimension = 15
        hidden_width = 192
        hidden_depth = 4
        activation = lambda x: jax.nn.leaky_relu(x, 1e-3)
        final_activation = lambda x: (1.0 - 0.0589) * jax.nn.sigmoid(x)

        self.interp_DL = np.logspace(-4, np.log10(15.0), 500)
        self.interp_z = z_at_value(
            Planck15.luminosity_distance, self.interp_DL * u.Gpc
        ).value

        super().__init__(
            model_weights,
            scaler,
            input_dimension,
            hidden_width,
            hidden_depth,
            activation,
            final_activation,
        )

    def _transform_parameters(
        self,
        m1_trials,
        m2_trials,
        a1_trials,
        a2_trials,
        cost1_trials,
        cost2_trials,
        z_trials,
        cos_inclination_trials,
        pol_trials,
        phi12_trials,
        ra_trials,
        sin_dec_trials,
    ):
        q = m2_trials / m1_trials
        eta = q / (1.0 + q) ** 2
        Mtot_det = (m1_trials + m2_trials) * (1.0 + z_trials)
        Mc_det = eta ** (3.0 / 5.0) * Mtot_det

        DL = jnp.interp(z_trials, self.interp_z, self.interp_DL)
        Mc_DL_ratio = Mc_det ** (5.0 / 6.0) / DL
        amp_factor_plus = jnp.log(
            (Mc_DL_ratio * ((1.0 + cos_inclination_trials**2) / 2)) ** 2
        )
        amp_factor_cross = jnp.log((Mc_DL_ratio * cos_inclination_trials) ** 2)

        # Effective spins
        chi_effective = (a1_trials * cost1_trials + q * a2_trials * cost2_trials) / (
            1.0 + q
        )
        chi_diff = (a1_trials * cost1_trials - a2_trials * cost2_trials) / 2.0

        # Generalized precessing spin
        Omg = q * (3.0 + 4.0 * q) / (4.0 + 3.0 * q)
        chi_1p = a1_trials * jnp.sqrt(1.0 - cost1_trials**2)
        chi_2p = a2_trials * jnp.sqrt(1.0 - cost2_trials**2)
        chi_p_gen = jnp.sqrt(
            chi_1p**2
            + (Omg * chi_2p) ** 2
            + 2.0 * Omg * chi_1p * chi_2p * jnp.cos(phi12_trials)
        )

        return jnp.array(
            [
                amp_factor_plus,
                amp_factor_cross,
                Mc_det,
                Mtot_det,
                eta,
                q,
                DL,
                ra_trials,
                sin_dec_trials,
                jnp.abs(cos_inclination_trials),
                jnp.sin(pol_trials % np.pi),
                jnp.cos(pol_trials % np.pi),
                chi_effective,
                chi_diff,
                chi_p_gen,
            ]
        )

    def predict(self, input_parameter_dict):
        # Copy so that we can safely modify dictionary in-place
        parameter_dict = input_parameter_dict.copy()

        # Check input
        parameter_dict = self.check_input(parameter_dict)

        features = jnp.array(
            [
                parameter_dict["mass_1"],
                parameter_dict["mass_2"],
                parameter_dict["a_1"],
                parameter_dict["a_2"],
                parameter_dict["cos_theta_1"],
                parameter_dict["cos_theta_2"],
                parameter_dict["redshift"],
                parameter_dict["cos_inclination"],
                parameter_dict["polarization_angle"],
                parameter_dict["phi_12"],
                parameter_dict["right_ascension"],
                parameter_dict["sin_declination"],
            ]
        )

        return self.__call__(features)
