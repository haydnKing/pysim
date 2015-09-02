"""Simulate the production and degredation of a fluorescent reporter"""

import pysim
import matplotlib.pyplot as plt

class ArsOperonModel(pysim.ODEGrowth):
	def __init__(self):
		super().__init__(pysim.GompertzGrowth(),
										 [
											('pArs', 1.), #promoter availability
											('Ars_mRNA', 0.), #mRNA
											('ArsR', 0.), 
											('ArsB', 0.), 
											('ArsR2', 0.), 
											('pArs_ArsR2', 0),
											('As_ext', 0.),
											('As_int', 0),
											('ArsR_As', 0),
										 ],
										 ['pArs','pArs_ArsR2', 'As_ext'],
										 k_transcription = 0.03,
										 k_translation = 0.07,
										 k_deg_mRNA = 0.01,
										 k_deg_prot = 0.001,
										 k_pArs_ArsR_fw = 1.0,
										 k_pArs_ArsR_rv = 0.01,
										 k_ArsR_ArsR_fw = 1.0,
										 k_ArsR_ArsR_rv = 0.01,
										 k_ArsR_As_fw = 1.0,
										 k_ArsR_As_rv = 0.01,
										 k_As_in = 0.0001,
										 k_As_out = 0.5)

	def define_reactions(self):
		return [
				self.build_reaction('pArs -> pArs + Ars_mRNA', 'k_transcription'),
				self.build_reaction('Ars_mRNA -> ', 'k_deg_mRNA'),
				self.build_reaction('Ars_mRNA -> Ars_mRNA + ArsR', 'k_translation'),
				self.build_reaction('Ars_mRNA -> Ars_mRNA + ArsB', 'k_translation'),
				self.build_reaction('ArsR -> ', 'k_deg_prot'),
				self.build_reaction('ArsB -> ', 'k_deg_prot'),
				self.build_reaction('2 ArsR <-> ArsR2', 'k_ArsR_ArsR_fw', 'k_ArsR_ArsR_rv'),
				self.build_reaction('ArsR2 -> ArsR', 'k_deg_prot'),
				self.build_reaction('ArsR2 + pArs <-> pArs_ArsR2', 'k_pArs_ArsR_fw', 'k_pArs_ArsR_rv'),
				self.build_reaction('ArsR + As_int <-> ArsR_As', 'k_ArsR_As_fw', 'k_ArsR_As_rv'),
				self.build_reaction('As_ext -> As_int', 'k_As_in'),
				self.build_reaction('As_int + ArsB -> ArsB + As_ext', 'k_As_out'),
			]

if __name__ == '__main__':
	print("Running model...")
	model = ArsOperonModel()
	model.condition_sweep(750,
												timestep=0.01,
												condition_to_vary = 'As_ext',
												condition_values = [0., 10000., 100000.,1000000.])
	model.plot(species='ArsB')
	plt.show(block=True)

