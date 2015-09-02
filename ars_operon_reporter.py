import ars_operon
import matplotlib.pyplot as plt
import numpy as np

class ArsReporterModel(ars_operon.ArsOperonModel):
	def __init__(self):
		super().__init__()
		self._add_species([
			('pArs_R', 1,), #seconds pArs copy
			('pArs_R_ArsR2', 0),
			('R_RNA', 0,), #reporter rna
			('R', 0,), #unactivated reporter
			('R*', 0), #activated reporter
			])
		self.no_dilute_species.append('pArs_R')
		self.no_dilute_species.append('pArs_R_ArsR2')
		self._add_parameters({
			'reporter_translation':0.5,
			'reporter_activation':0.1,
			})

	def define_reactions(self):
		return super().define_reactions() + [
				self.build_reaction('pArs_R + ArsR2 <-> pArs_R_ArsR2', 
					'k_pArs_ArsR_fw', 'k_pArs_ArsR_rv'),
				self.build_reaction('pArs_R -> pArs_R + R_RNA', 'k_transcription'),
				self.build_reaction('R_RNA ->', 'k_deg_mRNA'),
				self.build_reaction('R_RNA -> R_RNA + R', 'k_translation'),
				self.build_reaction('R -> ', 'k_deg_prot'),
				self.build_reaction('R -> R*', 'reporter_activation'),
				self.build_reaction('R* ->', 'k_deg_prot'),
			]

if __name__ == '__main__':
	print("Running model")
	model = ArsReporterModel()
	model.plot_response_at_time('R*',
															500,
															'As_ext',
															np.linspace(0., 1000000., 25),
															timestep=0.01)
	plt.show(block=True)
