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

	def build_reaction(self, reaction_str, fwd_param_name, rev_param_name=None):
		"""Utility function to build reaction
		reaction_str = "[reactant_def]* reaction_type [reactant_def]*"
			where:
				reactant_def := + [stoichiometry=1] species_name
					nb. the first '+' may be omitted from a list of reactant_defs
							and at least one reactant_def section must be present
				reaction_type = -> | <->
					->: irreversable reaction (-> rev_param_name is None)
					<->: reversable reaction


		"""

		#parse reaction string
		lhs = []
		rhs = []
		left = True
		cur_stoic = None
		got_plus = False
		reaction_type = ''

		for token in reaction_str.split(' '):
			#ignore ''
			if not token:
				continue
			#if we find reaction marker
			elif token in ('->', '<->'):
				#can't be part way through a reactant
				if cur_stoic is not None:
					raise ValueError('Expected species_name after \'{}\''.format(cur_stoic))
				#can't be expecting another reactant
				elif got_plus:
					raise ValueError('Expected species_name or stoichiometry after \'+\'')
				#can't have already had a reaction marker
				elif not left:
					raise ValueError('Unexpected \'{}\' after \'{}\''.format(token, reaction_type))
				else:
					reaction_type = token
					left=False
			#if we find a plus
			elif token == '+':
				#can't have already seen one
				if got_plus:
					raise ValueError('Unexpected \'+\' after \'+\'')
				#can't be expecting a species_name
				elif cur_stoic is not None:
					raise ValueError('Expected species_name after \'{}\', not \'+\''.format(cur_stoic))
				else:
					got_plus = True
			#if stoichiometry
			elif is_int(token):
				#must be positive
				if int(token)<=0:
					raise ValueError('stoichiometry value must be positive ({}<=0)'.format(int(token)))
				#must not have already seen a stoichiometry token
				elif cur_stoic is not None:
					raise ValueError('unexpected {} after stoichiometry {}'.format(int(token), cur_stoic)) 
				else:
					cur_stoic = int(token)
			elif token in self.species_names:
				#if there was no plus and this isn't the first species
				if not got_plus and len((lhs if left else rhs)) > 0:
					raise ValueError('expected \'+\' before species name \'{}\''.format(species_name))
				else:
					(lhs if left else rhs).append((cur_stoic if cur_stoic else 1, token))
					cur_stoic = None
					got_plus = False
			else:
				raise ValueError('unexpected token \'{}\''.format(token))

		
		if left or reaction_type == '':
			raise ValueError('Did not find reaction sign \'->\' or \'<->\'')

		if reaction_type == '->':
			if rev_param_name is not None:
				raise ValueError('Reverse parameter given in irreversable reaction')
			return Reaction(self.species_names, lhs, rhs, self._curr_params[fwd_param_name])
		
		return Reaction(self.species_names, 
										lhs, 
										rhs, 
										self._curr_params[fwd_param_name],
										self._curr_params[rev_param_name])



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

