from scipy.integrate import odeint
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import copy, itertools

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

class Reaction:
	def __init__(self, species_names, lhs, rhs, k_fw, k_rv=0):
		"""Define the reaction.
		Species names is an ordered list of the species names
		lhs and rhs should be a list of (stoichiometry, species_name) tuples
		k_fw and k_rv are the forward and reverse rate constants
		"""
		self.k_fw = k_fw
		self.k_rv = k_rv
		self.species_names = species_names

		#setup stochiometry arrays
		self.stoic_l = np.zeros(len(species_names))
		self.stoic_r = np.zeros(len(species_names))

		for stoic, name in lhs:
			self.stoic_l[species_names.index(name)] = stoic
	
		for stoic, name in rhs:
			self.stoic_r[species_names.index(name)] = stoic	

	def get_rates(self, y):
		K_fw = (-self.k_fw * np.product(np.power(y,self.stoic_l)) +
						self.k_rv * np.product(np.power(y,self.stoic_r)))
		
		#print("return {} * ({} - {})".format(K_fw, self.stoic_l, self.stoic_r))
		return K_fw * (self.stoic_l - self.stoic_r)

	def __str__(self):
		l = []
		for species, stoic in zip(self.species_names, self.stoic_l):
			if stoic > 0:
				if stoic == 1:
					l.append(species)
				else:
					l.append("{} {}".format(stoic, species))


		r = []
		for species, stoic in zip(self.species_names, self.stoic_r):
			if stoic > 0:
				if stoic == 1:
					r.append(species)
				else:
					r.append("{} {}".format(stoic, species))

		return '{} {} {}'.format(' + '.join(l), 
														 '->' if (self.k_rv==0) else '<->',
														 ' + '.join(r))

class ODEModel:
	def __init__(self, species, **kwargs):
		#names must be unique
		self.species_names = []
		self.initial_values = []
		for name, value in species:
			if name in self.species_names:
				raise ValueError('species names must be unique,'+ 
					' \'{}\' repeated'.format(name))
			self.species_names.append(name)
			self.initial_values.append(value)

		self.parameters = {}
		for k,v in kwargs.items():
			self.parameters[k] = v

		self.data=[]


		reactions = self.define_reactions(self.species_names, self.parameters)
		for r in reactions:
			print(r)

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

	def define_reactions(self, names, params):
		pass

	def _get_system_fn(self, params):
		reactions = self.define_reactions(self.species_names, params)
		def f(y, t):
			ret = np.zeros(len(y))
			for r in reactions:
				ret += r.get_rates(y)
		#	print(ret)
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

	def simulate_many(self,
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




