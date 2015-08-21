from pysim.odemodel import ODEModel

import numpy as np, matplotlib.pyplot as plt

class ExponentialGrowth:
	def __init__(self, double_time, t0 = 1):
		self.growth_rate = np.log(2) / double_time
		self.t0 = t0
	
	def mu(self, t):
		return self.growth_rate

	def cell_count(self, t):
		return self.t0 * np.exp(t * self.growth_rate)

class GompertzGrowth:
	def __init__(self, 
							 OD_0=0.1,
						 	 carrying_capacity=2.15, 
						 	 lag_time=111., 
							 maximal_growth_rate=0.0172):
		self.OD_0 = OD_0
		self.K = carrying_capacity
		self.l = lag_time
		self.mgr = maximal_growth_rate

	def cell_count(self, t):
		f = (self.mgr*np.exp(1.)/self.K)*(self.l - t) + 1
		return self.OD_0 * np.exp(self.K * np.exp( -np.exp(f)))

	def d_OD(self, t):
		f = (self.mgr*np.exp(1.)/self.K)*(self.l - t) + 1
		return (self.OD_0 * np.exp(self.K * np.exp( -np.exp(f))) * 
				np.exp(-np.exp(f)) * 
				np.exp(f) * self.mgr * np.exp(1))		

	def mu(self, t):
		return self.d_OD(t) / self.cell_count(t)

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


	def plot(self, species, ax=None):
		if ax is None:
			fig = plt.figure()
			ax1 = fig.add_subplot(211)
			super().plot(species, ax=ax1)

			ax2 = fig.add_subplot(212, sharex=ax1)
			time = self.data[0].df.index
			ax2.plot(time, self.growth_model.cell_count(time))

			ax2.set_ylabel("Cell Count")

	def save_data(self, start_cond, params, df):
		for species in self.species_names:
			if species not in self.no_dilute_species:
				df[species] = df[species].multiply(self.growth_model.cell_count(np.array(df.index)))
		super().save_data(start_cond, params, df)

