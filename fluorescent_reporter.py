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

	def define_reactions(self):
		return [
				self.build_reaction('promoter -> promoter + mRNA', 'k_transcription'),
				self.build_reaction('mRNA -> ', 'k_deg_mRNA'),
				self.build_reaction('mRNA -> mRNA + R', 'k_translation'),
				self.build_reaction('R -> ', 'k_deg_prot'),
				self.build_reaction('R -> R*', 'k_activation'),
				self.build_reaction('R* -> ', 'k_deg_prot'),
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

