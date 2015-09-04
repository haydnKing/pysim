import numpy as np, matplotlib.pyplot as plt, pandas as pd
from ars_operon_reporter import ArsReporterModel

class ConstitutiveArsRModel(ArsReporterModel):
	def __init__(self):
		super().__init__()
		self._add_species([
			('constArsR_RNA', 0,),
			])
		self._add_parameters({
			'k_constArsR': 0.1,
			})

	def define_reactions(self):
		return super().define_reactions() + [
				self.build_reaction('-> constArsR_RNA', 'k_constArsR'),
				self.build_reaction('constArsR_RNA -> ', 'k_deg_mRNA'),
				self.build_reaction('constArsR_RNA -> constArsR_RNA + ArsR', 
					'k_translation')
			]

if __name__ == '__main__':
	print("Running Model")
	model = ConstitutiveArsRModel()
	
	constArsR_values = [0.0, 0.01, 0.03]
	As_values = np.linspace(0,1000000, 25)

	df = pd.DataFrame(index=pd.Index(As_values, name="Arsenic"),
										columns=constArsR_values,
										data=np.zeros((len(As_values), len(constArsR_values))))
	for v in constArsR_values:
		model.set_parameter('k_constArsR', v)
		out = model.get_response_at_time(500,
															 			 'As_ext',
															 			 As_values,
															 			 timestep=0.01)
		df.loc[:,v] = out.loc[:,'R*']

	df.plot()
	plt.show(block=True)
