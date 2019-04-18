import os
import json

from itertools import product,combinations

augmentation_list = ['FlipLR', 'FlipUD', 'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 'CropBilinear', 'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout', 'Blur', 'Smooth']

augmentation_codes = {'FlipLR':"H", 'FlipUD':"V", 'AutoContrast':"A", 'Equalize':"E", 'Invert':"I", 'Rotate':"R", 'Posterize':"P", 'CropBilinear':"L", 'Solarize':"Z", 'Color':"U", 'Contrast':"C", 'Brightness':"N", 'Sharpness':"S", 'ShearX':"X", 'ShearY':"Y", 'TranslateX':"x", 'TranslateY':"y", 'Cutout':"O", 'Blur':"B", 'Smooth':"M"}

code_to_augmentation = {}
for k in augmentation_codes:
	code_to_augmentation[augmentation_codes[k]] = k

letter_list = code_to_augmentation.keys()
letter_list.sort()
print(letter_list)

def LoadPolicies(policiy_files, policy_dir_path):
	policies = []

	for policy_file in policiy_files:
		policy_file_path = os.path.join(policy_dir_path,policy_file)

		with open(policy_file_path, "r") as f:
			policies.append( json.loads(f.read()) )

	return policies


def CreateStructureStringFromPolicy(policy):
	policy_structure_string = ""
	for sub_policy in policy:
		for technique in sub_policy:
			# policy_structure_string += str(augmentation_list.index(technique[0]))
			policy_structure_string +=augmentation_codes[technique[0]]
			# policy_structure_string += "-"
		# policy_structure_string = policy_structure_string[:-1] +"_"

	# return policy_structure_string[:-1]
	return policy_structure_string[:]


def CreateBaseGenomes(policies):
	genomes = {}
	policy_dict = {}

	for policy in policies:
		policy_dict[policy["policy"]["id"]] = policy

		genome = CreateStructureStringFromPolicy(policy["policy"]["policy"])
		if(genome not in genomes):
			genomes[genome] = []

		genomes[genome].append(policy["policy"]["id"])

	return genomes


def GenerateMaskedGenomeString(base_genome_string,mask):
	string_list = list(base_genome_string)
	
	for i in range(len(mask)):
		if(mask[i]) == 1:
			string_list[i] = "?"

	return "".join(string_list)


def GenerateMaskedGenomeStringFromMaskIndex(base_genome_string,mask_indexs):
	string_list = list(base_genome_string)
	
	for i in mask_indexs:
		string_list[i] = "?"

	return "".join(string_list)


if __name__ == '__main__':
	num_sub_policies = 25
	num_techniques_per_sub_policy = 2

	generation_data_dir = ""
	policy_data_dir = "/media/harborned/ShutUpN/repos/final_year_project/genetic_augment/policies"

	experiment_name = "AutoAugmentBasedPopulation_exp_0002_120e_10p_25-2"

	experiment_policies = [policy_file for policy_file in os.listdir(policy_data_dir) if policy_file[:len(experiment_name)] == experiment_name]

	
	policies = LoadPolicies(experiment_policies, policy_data_dir)
	
	base_genomes = CreateBaseGenomes(policies)


	# for g in base_genomes.keys():
	# 	print(g)

	print(len(base_genomes.keys()))

	genomes = {}

	genomes[0] = base_genomes


	string_length = num_sub_policies * num_techniques_per_sub_policy
	max_string_i = string_length - 1

	for i in range(1,max_string_i):
		genomes[i] = {}

	max_wildcards = 4
	masks = product(*[[0,1]]*string_length)
	
	completed_masks = 0
	for i in range(1, max_wildcards+1):
		for mask_indexs in combinations( range(string_length), i ):
		 # = [1 if i in x else 0 for i in xrange(string_length)]
	# for mask in masks:
		# i = sum(mask)
			for base_genome_string in base_genomes:
				generated_genome = GenerateMaskedGenomeStringFromMaskIndex(base_genome_string,mask_indexs)

				if(generated_genome not in genomes[i]):
					genomes[i][generated_genome] = []

				genomes[i][generated_genome] += base_genomes[base_genome_string]					
			

			completed_masks +=1
			if(completed_masks % 1000 == 0):
				print(completed_masks)

	

	output_file = "generation_genome_data.json"

	with open(output_file,"w") as f:
		f.write(json.dumps(genomes,indent=4)) 

	for i in range(max_string_i):
		print(str(i))
		print(len(genomes[i]))

		# genome_keys = genomes[i].keys()
		# genome_keys.sort()
		# for genome in genome_keys:
		# 	print(genome)
			# for policy in genomes[i][genome]:
				#print("\t" + policy)
		print("___\n")


