import sys
import os

import json

print("Import Policy Functions")
from GeneticAugment import PolicyJsonToChromosome, CreateSpeciesAttributeDict, CreateRandomChromosome, CreateAllNeighbours, EvaluatePopulation, ArccaParallel, LocalSequential,TrainWithPolicyFitness, LogGeneration
from evaluator import augmentation_transforms
print("Importing Complete")



def LookUpKnownFitnessOfNeighbours(neighbours):
	#TODO: filter neighbours that actually need evaluating
	known_neighbours = []
	unknown_neighbours = neighbours
	return known_neighbours, unknown_neighbours


def MergeNeighbourFitness(known_neighbours, new_neighbour_fitness):
	#TODO: merge newly found fitnesses of neighbours with already known fitness of neighbours 
	return new_neighbour_fitness

        
if(__name__ == "__main__"):
	train_remote = True
	autoaugment_based_initial_policy = True

	data_path = "/media/harborned/ShutUpN/datasets/cifar/cifar-10-batches-py"
	if(len(sys.argv) > 1):
		data_path = sys.argv[1]
	experiment_attributes = {
		"num_epochs":60
		,"data_path":data_path
		,"dataset":"cifar10"
		,"model_name":"wrn"
		,"use_cpu":0
		,"clean_directories":True
		,"tabu_list_size": 100
		,"num_steps":10
		,"max_jobs":10
		
	}
	experiment_attributes["experiment_id"] = "tabu_fTransf_AutoAugmentBasedPopulation_exp_0001_"+str(experiment_attributes["num_epochs"])+"e_"+str(experiment_attributes["tabu_list_size"])+"ls_25-2"
	

	augmentation_list = list(augmentation_transforms.TRANSFORM_NAMES)
	augmentation_list = list(augmentation_transforms.FILTERED_TRANSFORM_NAMES)
	
	experiment_attributes["augmentation_list"] = augmentation_list
    

	if(train_remote):
		print("Training Remotely")
		experiment_attributes["population_evaluation_function"] = ArccaParallel
	else:
		print("Training Locally")
		experiment_attributes["population_evaluation_function"] = LocalSequential



	num_techniques_per_sub_policy = 2
	num_sub_policies_per_policy = 25

	population_size = 1

	species_attributes = CreateSpeciesAttributeDict(population_size,len(augmentation_list), num_techniques_per_sub_policy, num_sub_policies_per_policy)

	experiment_attributes["species_attributes"] = species_attributes



	experiment_attributes["local_ga_directory"] = "/media/harborned/ShutUpN/repos/final_year_project/genetic_augment"
	experiment_attributes["remote_ga_directory"] = "/home/c.c0919382/fyp_scw1427/genetic_augment"


	#initialise tabu list & tabu_list_size
	tabu_list = []
	

	#choose start policy
	print("Initialising Starting Policy")
	current_policy = None

	if(autoaugment_based_initial_policy):
		# paper_policy_path = "policies/tabu_AutoAugmentBasedPopulation_exp_0001_20e_100ls_25-2_00001_00850.json"
		paper_policy_path = "autoaugment_paper_cifar10.json"

		paper_policy_json = None

		with open(paper_policy_path, "r") as f:
			policy_string = f.read()
			paper_policy_json = json.loads(policy_string)

		current_policy = PolicyJsonToChromosome(paper_policy_json,augmentation_list,species_attributes)
		# best_policy_id = "paper_policy_20epoch"
		# best_accuracy = 0.62

	else:
		current_policy = CreateRandomChromosome(species_attributes)

	#evaluate starting policy 
	population_fitness, best_policy_id, best_accuracy  = EvaluatePopulation([current_policy], TrainWithPolicyFitness, experiment_attributes["augmentation_list"], experiment_attributes, 0)
	
	#initialise global best policy tracking
	best_global_policy = current_policy
	best_global_policy_id = best_policy_id
	best_global_fitness = best_accuracy

	for step in range(1,experiment_attributes["num_steps"]+1):
		print("Current Best Policy:")
		print(best_policy_id)
		print("Current Best Fitness:")
		print(best_accuracy)
		print("")
		print("Generating Neighbours")
		neighbours = CreateAllNeighbours(current_policy, species_attributes)
		print(str(len(neighbours))+" neighbours generated")

		#evaluate neighbours
		print("")
		print("evaluating neighbours")
		
		#TODO: only get fitness of unknown neighbours
		# known_fitness_neighbours, unknown_fitness_neighbours = LookUpKnownFitnessOfNeighbours(neighbours)

		population_fitness, best_policy_id, best_accuracy  = EvaluatePopulation(neighbours, TrainWithPolicyFitness, experiment_attributes["augmentation_list"], experiment_attributes, step)
		
		# neighbour_fitness = MergeNeighbourFitness(known_fitness_neighbours, population_fitness)
		
		best_found_policy = [p for p in population_fitness if p[2] == best_policy_id][0]

		if(best_accuracy > best_global_fitness):
			print("New Best Policy Found")
			best_global_fitness = best_accuracy
			best_global_policy_id = best_policy_id
			best_global_policy = best_found_policy[0]
		
		#move to best neighbour not in tabu list
		
		#TODO: implement actual TABU list move function 
		# current_policy = NextPolicy(population_fitness,tabu_list)
		current_policy = best_found_policy[0]

		#update tabu list
		#TODO: implement TABU list update function 
		# UpdateTabuList(current_policy)

		fitness_vals = [x[1]for x in population_fitness]

		generation_stats = {
				"step":step
				,"max_fitness":max(fitness_vals)
				,"min_fitness":min(fitness_vals)
				,"average_fitness":sum(fitness_vals)/float(len(fitness_vals))
				,"best_fitness_so_far":best_global_fitness
				,"best_policy_so_far":best_global_policy_id
			}

		LogGeneration(experiment_attributes["experiment_id"],generation_stats)