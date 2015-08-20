"""Simulate the production and degredation of a fluorescent reporter"""

import pysim
import matplotlib.pyplot as plt

class FluorescentReporterModel(pysim.ODEGrowth):
	def __init__(self):
		super().__init__(pysim.GompertzGrowth(),
										 [
											('promoter', 1.), #promoter availability
											('mRNA', 0.), #mRNA
											('R', 0.), #unactivated protein
											('R*', 0.), #activated protein
										 ],
										 ['promoter',],
										 k_transcription = 0.2,
										 k_translation = 0.5,
										 k_deg_mRNA = 0.1,
										 k_deg_prot = 0.01,
										 k_activation = 0.1)

	def define_reactions(self, names, params):
		return [
				pysim.Reaction(names, 
											 ((1, 'promoter',),),
											 ((1, 'promoter',),(1, 'mRNA'),),
											 params['k_transcription']),
				pysim.Reaction(names,
											 ((1, 'mRNA',),),
											 (),
											 params['k_deg_mRNA']),
				pysim.Reaction(names,
											 ((1, 'mRNA',),),
											 ((1, 'mRNA',),(1, 'R'),),
											 params['k_translation']),
				pysim.Reaction(names,
											 ((1, 'R',),),
											 (),
											 params['k_deg_prot']),
				pysim.Reaction(names,
											((1, 'R'),),
											((1, 'R*'),),
											params['k_activation']),
				pysim.Reaction(names,
											((1, 'R*'),),
											(),
											params['k_deg_prot']),
				]

if __name__ == '__main__':
	print("Running model...")
	model = FluorescentReporterModel()
	model.simulate_many(750,
											timestep=0.01,
											parameter_to_vary = 'k_deg_prot',
											parameter_values = [0.1, 0.02, 0.01,])
	model.plot(species='R*')
	plt.show(block=True)

