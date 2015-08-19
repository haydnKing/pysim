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

	def system_fn(self, t, params, promoter, mRNA, R, R_star):
		return [0, #constant promoter
						params['k_transcription'] * promoter - params['k_deg_mRNA'] * mRNA, #mRNA
						params['k_translation'] * mRNA - params['k_deg_prot'] * R, #R
						params['k_activation'] * R - params['k_deg_prot'] * R_star, #R*
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

