from scipy.integrate import odeint
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import copy, itertools

from reaction import Reaction

class ODEData:
	def __init__(self, start_conditions, parameters, df):
		self.start_conditions = start_conditions
		self.parameters = parameters
		self.df = df

	def __str__(self):
		l = ["{} = {}".format(k,v) for k,v in 
					itertools.chain(self.start_conditions.items(),
													self.parameters.items())]
		
		return '\n'.join(l)

class ODEModel:
	def __init__(self, species, **kwargs):
		#names must be unique
		self.species_names = []
		self.initial_values = []
		self._add_species(species)

		self.parameters = {}
		self._add_parameters(kwargs)

		self.data=[]


	def _add_species(self, species):
		for name, value in species:
			if name in self.species_names:
				raise ValueError('species names must be unique,'+ 
					' \'{}\' repeated'.format(name))
			self.species_names.append(name)
			self.initial_values.append(value)

	def _add_parameters(self, param_dict):
		for k,v in param_dict.items():
			if k in self.parameters.keys():
				raise ValueError('parameter names must be unique, '+
						'\'{}\' repeated'.format(k))
			self.parameters[k] = v

	def is_species(self, species_name):
		return species_name in self.species_names

	def get_initial_value(self, species_name):
		return self.initial_values[self.species_names.index(species_name)]
	
	def set_initial_value(self, species_name, value):
		self.initial_values[self.species_names.index(species_name)] = value

	def get_parameter(self, parameter_name):
		return self.parameters[parameter_name]

	def set_parameter(self, parameter_name, parameter_value):
		if not parameter_name in self.parameters.keys():
			raise ValueError('\'{}\' is not a know parameter'.format(parameter_name))
		self.parameters[parameter_name] = parameter_value

	def define_reactions(self):
		pass



	def _get_system_fn(self, params):
		self._curr_params = params
		reactions = self.define_reactions()
		def f(y, t):
			ret = np.zeros(len(y))
			for r in reactions:
				ret += r.get_rates(y)
			return ret
		return f

	def save_data(self, start_cond, params, df):
		self.data.append(ODEData(start_cond, params, df))

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

		t_out = np.arange(0, end_time, timestep)
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
		self.save_data(initial_values,
									 params,
									 self._simulate(end_time, 
															 timestep,
															 initial_values,
															 params))

	def parameter_sweep(self,
											end_time,
											parameter_to_vary,
											parameter_values,
											timestep = 0.1):
		for value in parameter_values:
			self.save_data({}, 
										 {parameter_to_vary:value,},
										 self._simulate(end_time,
																		timestep,
																		{},
																		{parameter_to_vary:value,}))

	def condition_sweep(self,
											end_time,
											condition_to_vary,
											condition_values,
											timestep = 0.1):
		for value in condition_values:
			self.save_data({condition_to_vary:value,}, 
										 {},
										 self._simulate(end_time,
																		timestep,
																		{condition_to_vary:value,},
																		{}))

	def get_response_at_time(self,
													 end_time,
													 condition_to_vary,
													 condition_values,
													 timestep=0.1):
		df = pd.DataFrame(index=pd.Index(condition_values, name=condition_to_vary),
											columns=self.species_names,
											data=np.zeros((len(condition_values),
												len(self.species_names))))

		for value in condition_values:
			out = self._simulate(end_time, 
													 timestep, 
													 {condition_to_vary:value,}, 
													 {})
			df.loc[value,:] = out.iloc[-1,:]

		return df

	def plot_response_at_time(self,
														species_to_plot,
														end_time,
														condition_to_vary,
														condition_values,
														timestep=0.1):
		df = self.get_response_at_time(end_time, 
																	 condition_to_vary,
																	 condition_values, 
																	 timestep)

		df[species_to_plot].plot()


	def plot(self, species, ax=None):
		if self.data is None:
			raise ValueError('No simulation data to plot')

		if species not in self.species_names:
			raise ValueError('Unknown species \'{}\''.format(species))

		if ax is None:
			ax = plt.figure().gca()

		for odedata in self.data:
			ax.plot(odedata.df.index, odedata.df[species], label=str(odedata))

		ax.set_xlabel("Simulation Time")
		ax.set_ylabel("\'{}\'".format(species))

		ax.legend(loc=0)

		return ax

	def __str__(self):
		self._curr_params = self.parameters
		return '\n'.join(str(r) for r in self.define_reactions())


def is_int(value):
	try:
		int(value)
	except:
		return False
	return True

