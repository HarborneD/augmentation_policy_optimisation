import json
import os

data_path = "/media/harborned/ShutUpN/repos/final_year_project/genetic_augment/generation_viewer"
generations_data_path = "generation_genome_data.json"


generations_genome_json = None

print("loading genome data")
print("")
with open(os.path.join(data_path,generations_data_path), "r") as f:
	generations_genome_json = json.loads(f.read())
print("genome data loaded")
print("")


# "AutoAugmentBasedPopulation_exp_0002_120e_10p_25-2_00007_00000"

def GetGenerationFromPolicyName(policy_name):
	name_split = policy_name.split("_")

	return int(name_split[-2])

num_generations = 100

generation_data_dict = [{} for g in range(num_generations)] 

wild_card_keys = [int(w) for w in list(generations_genome_json.keys())]

wild_card_keys.sort()

for generations_genome_json_key_int in wild_card_keys[:]:
	generations_genome_json_key = str(generations_genome_json_key_int)
	print("")
	print(str(generations_genome_json_key_int+1) + " of " + str(len(wild_card_keys)))

	genomes = generations_genome_json[generations_genome_json_key].keys()
	genome_count = 0
	total_genomes = len(genomes)
	for genome in genomes:
		genome_count+=1
		print(str(genome_count) + " of " + str(total_genomes))
		

		policies = generations_genome_json[generations_genome_json_key][genome]

		for policy in policies:
			# print(policy)
			generation = GetGenerationFromPolicyName(policy)

			if generations_genome_json_key_int not in generation_data_dict[generation]:
				generation_data_dict[generation][generations_genome_json_key_int] = {}

			if genome not in generation_data_dict[generation][generations_genome_json_key_int]:
				generation_data_dict[generation][generations_genome_json_key_int][genome] = []

			generation_data_dict[generation][generations_genome_json_key_int][genome].append(policy)


output_path = "generation_sorted_policies.json"


print("Writing data to JSON")
with open(output_path , "w") as f:
	f.write(json.dumps(generation_data_dict))