import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import h5py
from astropy.cosmology import Planck15,z_at_value
import astropy.units as u
import warnings
import pickle
import pandas

class emulator():

    def __init__(self,trained_weights,scaler_params,input_size,hidden_layer_width,hidden_layer_depth,activation):

        # Instantiate neural network
        self.trained_weights = trained_weights
        self.nn = eqx.nn.MLP(in_size=input_size,
                            out_size=1,
                            depth=hidden_layer_depth,
                            width_size=hidden_layer_width,
                            activation=activation,
                            final_activation=jax.nn.sigmoid,
                            key=jax.random.PRNGKey(111))

        # Load trained weights and biases
        weight_data = h5py.File(self.trained_weights,'r')

        # Load scaling parameters
        with open(scaler_params,'rb') as f:
            scaler = pickle.load(f) 
            self.scaler = {'mean':scaler.mean_,'scale':scaler.scale_}

        # Define helper functions with which to access MLP weights and biases
        # Needed by `eqx.tree_at`
        def get_weights(i):
            return lambda t: t.layers[i].weight
        def get_biases(i):
            return lambda t: t.layers[i].bias

        # Loop across layers, load pre-trained weights and biases
        for i in range(hidden_layer_depth):

            if i==0:
                key = 'dense'
            else:
                key = 'dense_{0}'.format(i)

            self.nn = eqx.tree_at(get_weights(i), self.nn, weight_data['{0}/{0}/kernel:0'.format(key)][()].T)
            self.nn = eqx.tree_at(get_biases(i), self.nn, weight_data['{0}/{0}/bias:0'.format(key)][()].T)

    def __call__(self,x):
        return jax.vmap(self.nn)((x-self.scaler['mean'])/self.scaler['scale'])

    def _check_distance(self,parameter_dict):

        """
        Helper function to check the presence of required distance arguments, and augment input
        parameters with additional quantities as needed.
        """

        # Check for distance parameters
        allowed_distance_params = ['luminosity_distance','comoving_distance','redshift']
        if not any(param in parameter_dict for param in allowed_distance_params):
            raise RuntimeError("Missing distance parameter. Requires one of:",allowed_distance_params)
        elif sum(param in parameter_dict for param in allowed_distance_params)>1:
            raise RuntimeError("Multiple distance parameters present. Only one of the following allowed:",allowed_distance_params)

        # Augment
        if 'comoving_distance' in parameter_dict:
            parameter_dict['redshift'] = z_at_value(Planck15.comoving_distance,parameter_dict['comoving_distance']*u.Gpc).value
            parameter_dict['luminosity_distance'] = Planck15.luminosity_distance(parameter_dict['redshift']).to(u.Gpc).value

        elif 'luminosity_distance' in parameter_dict:
            parameter_dict['redshift'] = z_at_value(Planck15.luminosity_distance,parameter_dict['luminosity_distance']*u.Gpc).value

        elif 'redshift' in parameter_dict:
            parameter_dict['luminosity_distance'] = Planck15.luminosity_distance(parameter_dict['redshift']).to(u.Gpc).value

    def _check_masses(self,parameter_dict):

        """
        Helper function to check the presence of required mass arguments, and augment
        input parameters with additional quantities needed for prediction.
        """

        # Check that mass parameters are present
        required_mass_params = ['mass_1','mass_2']
        for param in required_mass_params:
            if param not in parameter_dict:
                raise RuntimeError("Must include {0} parameter".format(param))

        # Reshape for safety below
        parameter_dict['mass_1'] = np.reshape(parameter_dict['mass_1'],-1)
        parameter_dict['mass_2'] = np.reshape(parameter_dict['mass_2'],-1)

        # Derived mass parameters
        m1_det = parameter_dict['mass_1']*(1.+parameter_dict['redshift'])
        m2_det = parameter_dict['mass_2']*(1.+parameter_dict['redshift'])
        parameter_dict['m1_det'] = m1_det
        parameter_dict['m2_det'] = m1_det 
        parameter_dict['eta'] = m1_det*m2_det/(m1_det+m2_det)**2
        parameter_dict['q'] = m2_det/m1_det
        parameter_dict['total_mass_det'] = m1_det+m2_det
        parameter_dict['chirp_mass_det'] = parameter_dict['eta']**(3./5.)*parameter_dict['total_mass_det']

    def _check_spins(self,parameter_dict):

        """
        Helper function to check for the presence of required spin parameters
        and augment with additional quantities as needed.
        """

        # Check that required spin parameters are present
        required_spin_params = ['a_1','a_2']
        for param in required_spin_params:
            if param not in parameter_dict:
                raise RuntimeError("Must include {0} parameter".format(param))

        # Reshape for safety below
        parameter_dict['a_1'] = np.reshape(parameter_dict['a_1'],-1)
        parameter_dict['a_2'] = np.reshape(parameter_dict['a_2'],-1)

        # Check for optional parameters, fill in if absent
        if 'cos_theta_1' not in parameter_dict:
            warnings.warn("Parameter 'cos_theta_1' not present. Filling with random value from isotropic distribution.")
            parameter_dict['cos_theta_1'] = 2.*np.random.random(parameter_dict['a_1'].shape)-1.
        if 'cos_theta_2' not in parameter_dict:
            warnings.warn("Parameter 'cos_theta_2' not present. Filling with random value from isotropic distribution.")
            parameter_dict['cos_theta_2'] = 2.*np.random.random(parameter_dict['a_2'].shape)-1.
        if 'phi_12' not in parameter_dict:
            warnings.warn("Parameter 'phi_12' not present. Filling with random value from isotropic distribution.")
            parameter_dict['phi_12'] = 2.*np.pi*np.random.random(parameter_dict['a_1'].shape)

    def _check_extrinsic(self,parameter_dict):

        """
        Helper method to check required extrinsic parameters and augment as necessary.
        """

        if 'right_ascension' not in parameter_dict:
            warnings.warn("Parameter 'right_ascension' not present. Filling with random value from isotropic distribution.")
            parameter_dict['right_ascension'] = 2.*np.pi*np.random.random(parameter_dict['mass_1'].shape)

        if 'sin_declination' not in parameter_dict:
            warnings.warn("Parameter 'sin_declination' not present. Filling with random value from isotropic distribution.")
            parameter_dict['sin_declination'] = 2.*np.random.random(parameter_dict['mass_1'].shape)-1.

        if 'inclination' not in parameter_dict:
            if 'cos_inclination' not in parameter_dict:
                warnings.warn("Parameter 'inclination' or 'cos_inclination' not present. Filling with random value from isotropic distribution.")
                parameter_dict['cos_inclination'] = 2.*np.random.random(parameter_dict['mass_1'].shape)-1.
        else:
            parameter_dict['cos_inclination'] = np.cos(parameter_dict['inclination'])

        if 'polarization_angle' not in parameter_dict:
            warnings.warn("Parameter 'polarization_angle' not present. Filling with random value from isotropic distribution.")
            parameter_dict['polarization_angle'] = 2.*np.pi*np.random.random(parameter_dict['mass_1'].shape)

    def check_input(self,parameter_dict):

        # Convert from pandas table to dictionary, if necessary
        if type(parameter_dict)==pandas.core.frame.DataFrame:
            parameter_dict = parameter_dict.to_dict(orient='list')

        # Check parameters
        self._check_distance(parameter_dict)
        self._check_masses(parameter_dict)
        self._check_spins(parameter_dict)
        self._check_extrinsic(parameter_dict)

        return parameter_dict

class p_det_O3(emulator):

    def __init__(self):
        
        model_weights="./../trained_weights/job_90_weights.hdf5"
        scaler="./../trained_weights/job_90_input_scaler.pickle"
        input_dimension=15
        hidden_width=192
        hidden_depth=3
        activation=lambda x: jax.nn.leaky_relu(x,1e-3)

        super().__init__(model_weights,scaler,input_dimension,hidden_width,hidden_depth,activation)

    def predict(self,input_parameter_dict):

        # Copy so that we can safely modify dictionary in-place
        parameter_dict = input_parameter_dict.copy()
        
        # Check input
        parameter_dict = self.check_input(parameter_dict)

        # Compute additional derived parameters
        Mc_DL_ratio = parameter_dict['chirp_mass_det']**(5./6.)/parameter_dict['luminosity_distance']
        amp_factor_plus = np.log((Mc_DL_ratio*((1.+parameter_dict['cos_inclination']**2)/2.))**2)
        amp_factor_cross = np.log((Mc_DL_ratio*parameter_dict['cos_inclination'])**2)
        log_Mc = np.log(parameter_dict['chirp_mass_det'])
        log_Mtot = np.log(parameter_dict['total_mass_det'])
        log_D = np.log(parameter_dict['luminosity_distance'])
        abs_cos_inc = np.abs(parameter_dict['cos_inclination'])
        sin_pol = np.sin(parameter_dict['polarization_angle'])
        cos_pol = np.cos(parameter_dict['polarization_angle'])

        # Effective spins
        a1 = parameter_dict['a_1']
        a2 = parameter_dict['a_2']
        cos_theta1 = parameter_dict['cos_theta_1']
        cos_theta2 = parameter_dict['cos_theta_2']
        chi_effective = (a1*cos_theta1 + parameter_dict['q']*a2*cos_theta2)/(1.+parameter_dict['q'])
        chi_diff = (a1*cos_theta1 - a2*cos_theta2)/2.

        # Generalized precessing spin
        Omg = parameter_dict['q']*(3.+4.*parameter_dict['q'])/(4.+3.*parameter_dict['q'])
        chi_1p = parameter_dict['a_1']*np.sqrt(1.-parameter_dict['cos_theta_1']**2)
        chi_2p = parameter_dict['a_2']*np.sqrt(1.-parameter_dict['cos_theta_2']**2)
        chi_p_gen = np.sqrt(chi_1p**2 + (Omg*chi_2p)**2 + 2.*Omg*chi_1p*chi_2p*np.cos(parameter_dict['phi_12']))

        features = jnp.array([
            amp_factor_plus,
            amp_factor_cross,
            log_Mc,
            log_Mtot,
            parameter_dict['eta'],
            parameter_dict['q'],
            log_D,
            parameter_dict['right_ascension'],
            parameter_dict['sin_declination'],
            abs_cos_inc,
            sin_pol,
            cos_pol,
            chi_effective,
            chi_diff,
            chi_p_gen
            ])

        return self.__call__(features.T)


