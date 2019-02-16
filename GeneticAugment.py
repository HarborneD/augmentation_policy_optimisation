from evaluator import augmentation_transforms

from StochasticUniversalSampler import StochasticUniversalSampler

import numpy as np
import random
import math

import tensorflow as tf
import evaluator.evaluate_policies_without_flags
import os 
import sys
import time
import json

from test_without_flags import TrainWithPolicy

from ArccaGAFunctions import RemoteGATool


##POPULATION FITNESS CALCULATION FUNCTIONS

def LocalSequential(fitness_function, policies, augmentations, experiment_attributes):
    population_fitness = []

    for policy in policies:
        population_fitness.append( (policy[0], fitness_function(policy[0],augmentations, experiment_attributes, policy[1])) )

    return population_fitness


def ArccaParallel(fitness_function, policies, augmentations, experiment_attributes):
    remote_tool = RemoteGATool(experiment_attributes["local_ga_directory"],experiment_attributes["remote_ga_directory"])

    policy_ids = []
    chromosome_dict = {}
    for policy in policies:
        policy_ids.append(policy[1])
        chromosome_dict[policy[1]] = policy[0]
    
    for policy_id in policy_ids:
        remote_tool.SendPolicyFile(policy_id)

    remote_tool.StartGenerationTraining(policy_ids,experiment_attributes["num_epochs"])

    remote_tool.WaitForGenerationComplete()

    time.sleep(2)
    results = remote_tool.GetGenerationResults()



    population_fitness = []

    for result in results:
        policy_id = result["policy_id"]
        population_fitness.append( ( chromosome_dict[policy_id], result["test_accuracy"]) )

    return population_fitness


##CHROMOSOME FITNESS FUNCTIONS


def StorePoliciesAsJsons(chromosome, augmentations, experiment_attributes, chromosome_id):
    policy_directory = "policies"

    policy_list = PolicyChromosomeToSubPoliciesList(chromosome, augmentations, experiment_attributes["species_attributes"])

    policy_id = experiment_attributes["experiment_id"]+"_"+str(chromosome_id)

    StorePolicyAsJson(policy_id, policy_list, policy_directory)

    return policy_id


def TrainWithPolicyFitness(chromosome, augmentations, experiment_attributes, policy_id):
    num_epochs = experiment_attributes["num_epochs"]
    data_path = experiment_attributes["data_path"]
    dataset = experiment_attributes["dataset"]
    model_name = experiment_attributes["model_name"]
    use_cpu = experiment_attributes["use_cpu"]

    _, test_acc = TrainWithPolicy(policy_id, num_epochs, data_path, dataset=dataset, model_name=model_name, use_cpu=use_cpu)


    return test_acc

def RemoteTrainWithPolicyFitness(chromosome, augmentations, experiment_attributes, policy_id):
    num_epochs = experiment_attributes["num_epochs"]
    data_path = experiment_attributes["data_path"]
    dataset = experiment_attributes["dataset"]
    model_name = experiment_attributes["model_name"]
    use_cpu = experiment_attributes["use_cpu"]

    _, test_acc = TrainWithPolicy(policy_id, num_epochs, data_path, dataset=dataset, model_name=model_name, use_cpu=use_cpu)


    return test_acc


def TestFitnessWithPolicy(chromosome, augmentations, experiment_attributes, policy_id):
    policy = PolicyChromosomeToSubPoliciesList(chromosome, augmentations, experiment_attributes["species_attributes"])
    # print("Policy:")
    # for sp in policy:
    #     print(sp)

    # print("run training")

    return TestFitness2(chromosome, augmentations, experiment_attributes,policy_id)


# def CreateJob(policy, model, epochs, dataset):
#     pass


def TestFitness(chromosome, augmentations, experiment_attributes, policy_id):
    def EvaluateTechnique(technique):
        current_augmentation_local_i = technique[0].index(1.0)

        return technique[1] * (current_augmentation_local_i +1)

    policy_val = 0
    for s_i in range(experiment_attributes["species_attributes"]["num_sub_policies"]):
        techniques = []
        for t_i in range(experiment_attributes["species_attributes"]["num_technqiues"]):
            tech, _, _, _ = FetchTechniqueFromLocalIndex(chromosome,t_i,s_i,experiment_attributes["species_attributes"])
            techniques.append( tech )
        
        sub_policy_val = 10 - abs(2 - EvaluateTechnique(techniques[0]) - EvaluateTechnique(techniques[1]))    
        policy_val += sub_policy_val

    return policy_val


def TestFitness2(chromosome, augmentations, experiment_attributes, policy_id):
    ### fitness = sum( foreach subpolicy: foreach technique: technique_index )
    ### max = num_sub_policies * 2 * (num_augmentations-1)
    def EvaluateTechnique(technique):
        current_augmentation_local_i = technique[0].index(1.0)

        return current_augmentation_local_i
    
    policy_val = 0
    for s_i in range(experiment_attributes["species_attributes"]["num_sub_policies"]):
        techniques = []
        for t_i in range(experiment_attributes["species_attributes"]["num_technqiues"]):
            tech, _, _, _ = FetchTechniqueFromLocalIndex(chromosome,t_i,s_i,experiment_attributes["species_attributes"])
            techniques.append( tech )
        
        sub_policy_val = EvaluateTechnique(techniques[0]) + EvaluateTechnique(techniques[1])   
        policy_val += sub_policy_val

    return policy_val






###EVOLUTION FUNCTIONS
def CrossoverPolicy(chromosome_1,chromosome_2,prob_crossover,species_attributes):
    if(random.random() < prob_crossover):
        cross_i = random.randint(1,species_attributes["num_sub_policies"]) #perform crossover before this index
    
        second_half_1 = chromosome_1[cross_i*species_attributes["sub_policy_size"]:]
        second_half_2 = chromosome_2[cross_i*species_attributes["sub_policy_size"]:]

        chromosome_1 = chromosome_1[:cross_i*species_attributes["sub_policy_size"]] + second_half_2
        chromosome_2 = chromosome_2[:cross_i*species_attributes["sub_policy_size"]] + second_half_1
    
    return chromosome_1, chromosome_2


def MutateTechnique(chromosome, prob_technique_mutate,species_attributes):
    for s_i in range(species_attributes["num_sub_policies"]):
        for t_i in range(species_attributes["num_technqiues"]):
            if(random.random() < prob_technique_mutate):
                
                current_technique, t_global_i, t_probability_global_i, t_magnitude_global_i = FetchTechniqueFromLocalIndex(chromosome,t_i, s_i,species_attributes)
                current_augmentation_local_i = current_technique[0].index(1.0)

                new_augmentation_local_i = current_augmentation_local_i
                while(new_augmentation_local_i == current_augmentation_local_i):
                    new_augmentation_local_i = random.randint(0,species_attributes["num_augmentations"]-1)
                
                chromosome[t_global_i+current_augmentation_local_i] = 0
                chromosome[t_global_i+new_augmentation_local_i] = 1

    return chromosome

        
def MutateMagnitude(chromosome,prob_magnitude_mutate,species_attributes):
    for s_i in range(species_attributes["num_sub_policies"]):
        for t_i in range(species_attributes["num_technqiues"]):
            if(random.random() < prob_magnitude_mutate):
                global_magnitude_i = MapLocalMagnitudeIndexsToGlobalInChromosome(t_i, s_i, species_attributes)

                magnitude_possibilities = []
                current_magnitude = chromosome[global_magnitude_i]
                if(current_magnitude >= 0.1):
                    magnitude_possibilities.append(current_magnitude-0.1)

                if(current_magnitude <= 0.8):
                    magnitude_possibilities.append(current_magnitude+0.1)
                
                new_magnitude = random.choice(magnitude_possibilities)

                chromosome[global_magnitude_i] = new_magnitude
    
    return chromosome


def MutateProbability(chromosome,prob_probability_mutate,species_attributes):
    for s_i in range(species_attributes["num_sub_policies"]):
        for t_i in range(species_attributes["num_technqiues"]):
            if(random.random() < prob_magnitude_mutate):
                global_magnitude_i = MapLocalProbabilityIndexsToGlobalInChromosome(t_i, s_i, species_attributes)

                probability_possibilities = []
                current_probability = chromosome[global_magnitude_i]
                if(current_probability >= 0.1):
                    probability_possibilities.append(current_probability-0.1)

                if(current_probability <= 0.9):
                    probability_possibilities.append(current_probability+0.1)
                
                new_magnitude = random.choice(probability_possibilities)

                chromosome[global_magnitude_i] = new_magnitude
    
    return chromosome




###MAPPING and FORMATTING FUNCTIONS
def MapLocalAugmentationIndexsToGlobalInChromosome(augmentation_i, technique_i, sub_policy_i, species_attributes):
    return augmentation_i + (species_attributes["num_augmentations"]* technique_i) + (species_attributes["sub_policy_size"] * sub_policy_i)

def MapLocalProbabilityIndexsToGlobalInChromosome(technique_i, sub_policy_i, species_attributes):
    return  (species_attributes["sub_policy_size"] * sub_policy_i) + species_attributes["total_sp_one_hot_elements"] + technique_i 

def MapLocalMagnitudeIndexsToGlobalInChromosome(technique_i, sub_policy_i, species_attributes):
    return  (species_attributes["sub_policy_size"] * sub_policy_i) + species_attributes["total_sp_one_hot_elements"] + species_attributes["num_technqiues"] + technique_i 


def FetchTechniqueFromLocalIndex(chromosome, technique_i, sub_policy_i, species_attributes):
    augmentation_vector_start_i = MapLocalAugmentationIndexsToGlobalInChromosome(0, technique_i, sub_policy_i, species_attributes)

    probability_index = MapLocalProbabilityIndexsToGlobalInChromosome(technique_i, sub_policy_i, species_attributes)
    magnitude_index = MapLocalMagnitudeIndexsToGlobalInChromosome(technique_i, sub_policy_i, species_attributes)
    # print(technique_i, sub_policy_i)
    # print(chromosome)
    # print(magnitude_index)
    # print(augmentation_vector_start_i)

    return (chromosome[augmentation_vector_start_i:augmentation_vector_start_i+species_attributes["num_augmentations"]] ,chromosome[probability_index], chromosome[magnitude_index]), augmentation_vector_start_i, probability_index, magnitude_index


def FetchSubPolicyFromIndex(chromosome, sub_policy_i, species_attributes):
    sp_global_start_index = MapLocalAugmentationIndexsToGlobalInChromosome(0, 0, sub_policy_i, species_attributes)

    return chromosome[sp_global_start_index:sp_global_start_index+species_attributes["sub_policy_size"]], sp_global_start_index


def TechniqueToAugmentationTuple(technique,augmentations):
    augmentation_i = technique[0].index(1.0)

    return (augmentations[augmentation_i], technique[1], int(technique[2]*10))


def PolicyChromosomeToSubPoliciesList(chromosome, augmentations, species_attributes):
    sub_policies = []
    for s_i in range(species_attributes["num_sub_policies"]):
        selected_augmentations = []
        techniques = []
        for t_i in range(species_attributes["num_technqiues"]):
            tech, _, _, _ = FetchTechniqueFromLocalIndex(chromosome,t_i,s_i,species_attributes)
            techniques.append(tech[:])
        
        for technique in techniques:
            selected_augmentations.append(TechniqueToAugmentationTuple(technique,augmentations))
        sub_policies.append({"s_i":s_i, "augmentations":selected_augmentations[:]})    

    return sub_policies



def StorePolicyAsJson(policy_id, policy_list, policy_directory):
    policy_json = PolicyListToPolicyJSON(policy_list,policy_id)

    policy_json_path = os.path.join(policy_directory, policy_id+".json")
    with open(policy_json_path, "w") as f:
        f.write(policy_json)


def PolicyListToPolicyJSON(policy_list, policy_id):
    policy_dict = {
        "policy":{
        "id":policy_id,
        "policy":
            [[list(a) for a in p["augmentations"]] for p in policy_list]
        
        }
    }


    policy_json_string = json.dumps(policy_dict,indent=4)

    return policy_json_string
    



##GENERATE CHOMOSOME FUNCTIONS
def BuildBlankPolicyChromosome(species_attributes):
    # format: [ [one hot encoded technique]*num_technique + [probabilities]*num_technique + [magnitude]*num_technique ] *num_sub_policies
    # EG:   [1000000 0100000 0.1 0.6  0010000 0100000 0.0 0.5   0000001 0100000 0.7 0.1] (one hot encodings are also floats but represented as ints for readability)
    return [0.0] * (species_attributes["sub_policy_size"] * species_attributes["num_sub_policies"])   
    

def CreateRandomChromosome(species_attributes):
    chromosome = BuildBlankPolicyChromosome(species_attributes)
    for s_i in range(species_attributes["num_sub_policies"]):
        for t_i in range(species_attributes["num_technqiues"]):
            augmentation_i = random.randint(0,species_attributes["num_augmentations"]-1)
            chromosome[ MapLocalAugmentationIndexsToGlobalInChromosome(augmentation_i, t_i, s_i, species_attributes) ] = 1.0

            magnitude = random.randint(0,10)/10.0
            chromosome[MapLocalMagnitudeIndexsToGlobalInChromosome(t_i, s_i, species_attributes)] = magnitude
            
            probability = random.randint(0,9)/10.0
            chromosome[MapLocalProbabilityIndexsToGlobalInChromosome(t_i, s_i, species_attributes)] = probability
            
    return chromosome





###EVOLUTION STAGE FUNCTIONS
def EvaluatePopulation(population,fitness_function, augmentations, experiment_attributes, step):
    population_fitness = []

    policies = []
    for chromosome_i in range(len(population)):
        chromosome_id = format(step, '05d') + "_" + format(chromosome_i, '05d')
        chromosome = population[chromosome_i]

        policies.append( (chromosome, StorePoliciesAsJsons(chromosome, augmentations, experiment_attributes, chromosome_id)) )
    
    population_fitness = experiment_attributes["population_evaluation_function"](fitness_function, policies, augmentations, experiment_attributes)
    # for policy in range(len(policies)):
    #     population_fitness.append( (policy[0], fitness_function(policies[0],augmentations, experiment_attributes, policy[1])) )

    return population_fitness


def PerformCrossoverAndMutations(chromosome_1,chromosome_2, evolution_probabilities, species_attributes):
    #print("chromosome_1",chromosome_1)
    chromosome_1,chromosome_2 = CrossoverPolicy(chromosome_1,chromosome_2,evolution_probabilities["prob_crossover"],species_attributes)
    
    chromosome_1 = MutateTechnique(chromosome_1, evolution_probabilities["prob_technique_mutate"],species_attributes)
    chromosome_2 = MutateTechnique(chromosome_2, evolution_probabilities["prob_technique_mutate"],species_attributes)

    chromosome_1 = MutateProbability(chromosome_1,evolution_probabilities["prob_probability_mutate"],species_attributes)
    chromosome_2 = MutateProbability(chromosome_2,evolution_probabilities["prob_probability_mutate"],species_attributes)

    chromosome_1 = MutateMagnitude(chromosome_1,evolution_probabilities["prob_magnitude_mutate"],species_attributes)
    chromosome_2 = MutateMagnitude(chromosome_2,evolution_probabilities["prob_magnitude_mutate"],species_attributes)

    return chromosome_1[:],chromosome_2[:]


def EvolveSelected(selected_chromosomes,evolution_probabilities, species_attributes):
    random.shuffle(selected_chromosomes)

    evolved_chromosomes = []

    for i in range(0,len(selected_chromosomes),2):
        # print(selected_chromosomes[i])
        c_1, c_2 = PerformCrossoverAndMutations(selected_chromosomes[i][0],selected_chromosomes[i+1][0], evolution_probabilities, species_attributes)
        
        evolved_chromosomes.append(c_1[:])
        evolved_chromosomes.append(c_2[:])
    
    return evolved_chromosomes


def MoveToNextGeneration(population_fitness, evolution_probabilities, species_attributes, elitism_ratio=0.05):
    next_generation = []

    population_fitness = sorted(population_fitness, key=lambda x: x[1], reverse=True)
    #print(population_fitness[0])
    num_elite = int(math.ceil((species_attributes["population_size"]*elitism_ratio) / 2.) * 2)
    
    next_generation += [x[0] for x in population_fitness[:num_elite]]
    next_generation[0]
    # for n in next_generation:
    #     print(n)
    # print("++++")
    # print("")

    sampler = StochasticUniversalSampler(population_fitness)

    selected = sampler.GenerateSelections(species_attributes["population_size"] - len(next_generation))
    
    evolved = EvolveSelected(selected,evolution_probabilities, species_attributes)


    # for e in evolved:
    #     print(e)
    # print("eeeee")
    # print("")

    next_generation += evolved    

    return next_generation[:]
    



#ATTRIBUTE DICTIONARY FUNCTIONS
def CreateSpeciesAttributeDict(population_size, num_augmentations, num_technqiues_per_sub_policy, num_sub_policies_per_policy):
    return {
        "population_size": population_size,
        "num_augmentations":num_augmentations,
        "num_technqiues":num_technqiues_per_sub_policy,
        "num_sub_policies":num_sub_policies_per_policy,
        "sub_policy_size":(num_augmentations + 2)* num_technqiues_per_sub_policy,
        "total_sp_one_hot_elements":num_augmentations * num_technqiues_per_sub_policy
    }


def CreateProbabilitiesDict(prob_crossover,prob_technique_mutate,prob_probability_mutate,prob_magnitude_mutate):
    return {
        "prob_crossover": prob_crossover,
        "prob_technique_mutate": prob_technique_mutate,
        "prob_probability_mutate": prob_probability_mutate, 
        "prob_magnitude_mutate": prob_magnitude_mutate 
    }



###LOGGING
def LogGeneration(experiment_id,generation_stats):
    generation_string =""

    keys = ["step","max_fitness","min_fitness","average_fitness"]
    for k in keys:
        generation_string += str(generation_stats[k]) + ","
    
    generation_string = generation_string[:-1]+"\n"
    with open(experiment_id+".csv","a") as f:
        f.write(generation_string)




if(__name__ == "__main__"):
    train_remotely = False
    data_path = "/media/harborned/ShutUpN/datasets/cifar/cifar-10-batches-py"
    if(len(sys.argv) > 1):
        data_path = sys.argv[1]
    experiment_attributes = {
        "experiment_id":"test_remote_exp_0001_20e_10p_5-2"
        ,"num_epochs":1
        ,"data_path":data_path
        ,"dataset":"cifar10"
        ,"model_name":"wrn"
        ,"use_cpu":0
        
    }

    if(train_remotely):
        experiment_attributes["population_evaluation_function"] = ArccaParallel
    else:
        experiment_attributes["population_evaluation_function"] = LocalSequential

    augmentation_list = list(augmentation_transforms.TRANSFORM_NAMES)
    print("")
    print("number of augmentations: ", len(augmentation_list))
    print("")
    num_technqiues_per_sub_policy = 2
    num_sub_policies_per_policy = 5
    
    population_size = 4

    prob_crossover = 0.001
    prob_technique_mutate = 0.001
    prob_magnitude_mutate = 0.001
    prob_probability_mutate = 0.001


    num_evolution_steps = 100

    fitness_function = TrainWithPolicyFitness
    

    evolution_probabilities = CreateProbabilitiesDict(prob_crossover,prob_technique_mutate,prob_probability_mutate,prob_magnitude_mutate)

    species_attributes = CreateSpeciesAttributeDict(population_size,len(augmentation_list), num_technqiues_per_sub_policy, num_sub_policies_per_policy)
    
    experiment_attributes["species_attributes"] = species_attributes

    experiment_attributes["local_ga_directory"] = "/media/harborned/ShutUpN/repos/final_year_project/genetic_augment"
    experiment_attributes["remote_ga_directory"] = "/home/c.c0919382/fyp_scw1427/genetic_augment"

    population = []
    for p_i in range(population_size):
        population.append(CreateRandomChromosome(species_attributes))
    

    # for p in population:
    #     print(p)

    # print("bbbbb")


    for step in range(num_evolution_steps):
        print("____")
        print("Starting evolution step: " +str(step))
        print("")

        population_fitness = EvaluatePopulation(population,fitness_function, augmentation_list, experiment_attributes, step)
        
      
        # for p in population_fitness:
        #     print(p)

        # print("fffffff")
        # print("")

     

        fitness_vals = [x[1]for x in population_fitness]

        generation_stats = {
                "step":step
                ,"max_fitness":max(fitness_vals)
                ,"min_fitness":min(fitness_vals)
                ,"average_fitness":sum(fitness_vals)/float(len(fitness_vals))
            }

        LogGeneration(experiment_attributes["experiment_id"],generation_stats)

        if(step % 1==0):
            print("step:",step)
            print("Max Fitness:", generation_stats["max_fitness"])
            print("Min Fitness:", generation_stats["min_fitness"] )
            print("Average Fitness:", generation_stats["average_fitness"] )
            # for p in population_fitness:
            #     print(p)
            print("-----")
            print("")

            


        population = MoveToNextGeneration(population_fitness, evolution_probabilities, species_attributes, elitism_ratio=0.05)


# for p in population:
#     print(p)

# print("fffff")


print(population[0])
print(fitness_function(population[0], augmentation_list,experiment_attributes, "post_evolution_test_1"))