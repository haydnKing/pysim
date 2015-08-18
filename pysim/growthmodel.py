from pysim.odemodel import ODEModel

import numpy as np

class ExponentialGrowth:
	def __init__(self, double_time, t0 = 1):
		self.growth_rate = np.log(2) / double_time
		self.t0 = t0
	
	def mu(self, t):
		return self.growth_rate

	def cell_count(self, t):
		return t0 * np.exp(t * self.growth_rate)

class ODEGrowth(ODEModel):
	def __init__(self, growth_model, species, no_dilute, **kwargs):
		super().__init__(species, **kwargs)

		self.growth_model = growth_model
		self.no_dilute_species = []
		for s in no_dilute:
			if s in self.species_names:
				self.no_dilute_species.append(s)
			else:
				raise ValueError('unknown species \'{}\''.format(s))

	def _get_system_fn(self, params):
		orig_fn = super()._get_system_fn(params)
		def fn(y, t):
			v = orig_fn(y,t)
			for i,s in enumerate(self.species_names):
				if s not in self.no_dilute_species:
					v[i] = v[i] - self.growth_model.mu(t) * y[i]
			return v
		return fn

#plotting code, multiply by cell number

