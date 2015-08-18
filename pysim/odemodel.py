from scipy.integrate import odeint
import numpy as np, pandas as pd
from matplotlib import pyplot as plt


class ODEModel:
	def __init__(self, species_names, default_initial_values, **kwargs):
		#names must be unique
		self.species_names = []
		for name in species_names:
			if name in self.species_names:
				raise ValueError('species names must be unique,'+ 
					' \'{}\' repeated'.format(name))
			self.species_names.append(name)
		self.initial_values = np.array(default_initial_values)

		#species_names and default_initial_values should be the same length
		if len(self.species_names) != len(self.default_initial_values):
			raise ValueError('Should provide the same number of names as initial values')
		self.parameters = {}
		for k,v in kwargs.items():
			self.parameters[k] = v

	def is_species(self, species_name):
		return species_name in self.species_names

	def get_initial_value(self, species_name):
		return self.initial_values[self.species_names.index(species_name)]
	
	def set_initial_value(self, species_name, value):
		self.initial_values[self.species_names.index(species_name)] = value

	def get_parameter(self, parameter_name):
		return self.parameter_values(parameter_name)

	def set_parameter(self, parameter_name, parameter_value):
		if not parameter_name in self.parameters:
			raise ValueError('\'{}\' is not a know parameter'.format(parameter_name))
		self.parameter_values[parameter_name]

	def system_fn(self, t, params):
		pass

	def _get_system_fn(self, params):
		def f(y, t):
			return self.system_fn(t, params, *y)
		return f

	def _simulate(self, 
								end_time, 
								timestep = 0.1,
								initial_values = {},
								params = {}):
		#copy species
		sim_vars = copy.copy(self.initial_values)
		for k,v in initial_values.items():
			#check for duff initial_values
			if k not in self.species_names:
				raise ValueError('Unknown species \'{}\''.format(k))
			sim_vars[self.species_names.index(k)] = v 

		#copy parameters
		sim_params = copy.copy(self.parameters)
		for k,v in params.items():
			#check for unknown parameters
			if k not in self.parameters:
				raise ValueError('Unknown parameter \'{}\''.format(k))
			sim_params[k] = v

		t_out = np.arrange(0, end_time, timestep)
		y_out = odeint(self._get_system_fn(sim_params),
									 sim_vars,
									 t_out)

		return pd.DataFrame(data=y_out,
												index=t_out,
												columns=self.species_names)

	def simulate(self, 
							 end_time, 
							 timestep = 0.1,
							 initial_values = {},
							 params = {}):
		self.data = self._simulate(end_time, 
															 initial_values,
															 timestep,
															 params)
		return self.data

	def plot(self, species=None):
		if not self.data:
			raise ValueError('No simulation data to plot')

		if species == None:
			species = self.species_names

		for s in species:
			if s not in self.species_names:
				raise ValueError('Unknown species \'{}\''.format(s))

		fig = plt.figure()
		ax = fig.gca()

		ax.plot(self.data.index, self.data.loc[species], label=species)

		ax.set_xlabel("Simulation Time")
		ax.set_ylabel("Species Amount")

		ax.legend(loc=0)

		fig.show()





