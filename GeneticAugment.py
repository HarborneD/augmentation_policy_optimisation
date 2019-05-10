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
import shutil

from test_without_flags import TrainWithPolicy

from ArccaGAFunctions import RemoteGATool


##POPULATION FITNESS CALCULATION FUNCTIONS

def LocalSequential(fitness_function, policies, augmentations, experiment_attributes):
    population_fitness = []

    best_policy_id = None
    best_accuracy = 0
    for policy in policies:
        population_fitness.append( (policy[0], fitness_function(policy[0],augmentations, experiment_attributes, policy[1])) )
        if(population_fitness[-1][1] > best_accuracy):
            best_accuracy = population_fitness[-1][1]
            best_policy_id = policy[1]
    
    if(experiment_attributes["clean_directories"]):
        LocalCleanDirectoriesAndStoreCurrentGen([p[1] for p in policies])

    return population_fitness, best_policy_id, best_accuracy


def ArccaParallel(fitness_function, policies, augmentations, experiment_attributes, repeats_per_policy=3):
    #fitness_function is not used and is included to support other forms of evaluation function
    max_jobs = 10
    if("max_jobs" in experiment_attributes):
        max_jobs = experiment_attributes["max_jobs"]

    
    num_batches = int(math.ceil(len(policies) / float(max_jobs)))

    clear_checkpoint_limit = 10
    always_clear_checkpoints = len(policies) > clear_checkpoint_limit

    cumulative_results = {}

    remote_tool = RemoteGATool(experiment_attributes["local_ga_directory"],experiment_attributes["remote_ga_directory"])

    policy_ids = []
    chromosome_dict = {}
    for policy in policies:
        policy_ids.append(policy[1])
        chromosome_dict[policy[1]] = policy[0]
    
    for policy_id in policy_ids:
        remote_tool.SendPolicyFile(policy_id)
        cumulative_results[policy_id] = 0
    
    best_single_trial_accuracy = 0
    best_single_trial_policy_id = ""

    last_job_i = -1
    while((last_job_i+1) < len(policies)):
    # for batch_i in range(num_batches):
        active_jobs = len(remote_tool.arcca_tool.GetActiveJobs())
        jobs_in_batch = max_jobs - active_jobs

        # print("Batch "+str(batch_i+1)+" of "+str(num_batches))
        # batch_start = batch_i*max_jobs
        # batch_end = min((batch_i+1)*max_jobs , len(policies))
        
        batch_start = last_job_i+1
        batch_end = min(batch_start+jobs_in_batch , len(policies))
        last_job_i = (batch_end-1)

        batch_policy_ids = policy_ids[batch_start:batch_end]
        print("Batch Index Range(inc.): "+str(batch_start)+" to "+str(last_job_i))
        

        for repeat_i in range(repeats_per_policy):
            remote_tool.StartGenerationTraining(batch_policy_ids,experiment_attributes["num_epochs"])

            remote_tool.WaitForGenerationComplete()

            time.sleep(2)
            results = remote_tool.GetGenerationResults()

            for result in results:
                policy_id = result["policy_id"]
                cumulative_results[policy_id] += result["test_accuracy"]

                if(result["test_accuracy"] > best_single_trial_accuracy):
                    best_single_trial_accuracy = result["test_accuracy"]
                    best_single_trial_policy_id = policy_id
                    print("New Best Single Policy:")
                    print(best_single_trial_policy_id + ": " + str(best_single_trial_accuracy))

            if(always_clear_checkpoints or (repeat_i != (repeats_per_policy-1))):
                remote_tool.CleanCheckpoints(batch_policy_ids)
    

    results = []
    for k in cumulative_results:
        results.append({"policy_id":k,"test_accuracy": cumulative_results[k] / float(repeats_per_policy)})


    population_fitness = []

    best_policy_id = None
    best_accuracy = 0
    for result in results:
        policy_id = result["policy_id"]
        population_fitness.append( ( chromosome_dict[policy_id], result["test_accuracy"], policy_id) )
        if(result["test_accuracy"] > best_accuracy):
            best_accuracy = result["test_accuracy"]
            best_policy_id = policy_id

    if( (not always_clear_checkpoints) and experiment_attributes["clean_directories"] ):
        remote_tool.CleanDirectoriesAndStoreCurrentGen(policy_ids)

    return population_fitness, best_policy_id, best_accuracy


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
        for t_i in range(experiment_attributes["species_attributes"]["num_techniques"]):
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
        for t_i in range(experiment_attributes["species_attributes"]["num_techniques"]):
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
        for t_i in range(species_attributes["num_techniques"]):
            if(random.random() < prob_technique_mutate):
                
                current_technique, t_global_i, _, _ = FetchTechniqueFromLocalIndex(chromosome,t_i, s_i,species_attributes)
                current_augmentation_local_i = current_technique[0].index(1.0)

                new_augmentation_local_i = current_augmentation_local_i
                while(new_augmentation_local_i == current_augmentation_local_i):
                    new_augmentation_local_i = random.randint(0,species_attributes["num_augmentations"]-1)
                
                chromosome[t_global_i+current_augmentation_local_i] = 0.0
                chromosome[t_global_i+new_augmentation_local_i] = 1.0

    return chromosome

        
def MutateMagnitude(chromosome,prob_magnitude_mutate,species_attributes):
    for s_i in range(species_attributes["num_sub_policies"]):
        for t_i in range(species_attributes["num_techniques"]):
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
        for t_i in range(species_attributes["num_techniques"]):
            if(random.random() < prob_magnitude_mutate):
                global_probability_i = MapLocalProbabilityIndexsToGlobalInChromosome(t_i, s_i, species_attributes)

                probability_possibilities = []
                current_probability = chromosome[global_probability_i]
                if(current_probability >= 0.1):
                    probability_possibilities.append(current_probability-0.1)

                if(current_probability <= 0.9):
                    probability_possibilities.append(current_probability+0.1)
                
                new_probability = random.choice(probability_possibilities)

                chromosome[global_probability_i] = new_probability
    
    return chromosome


def CreateAllNeighbours(chromosome,species_attributes):
    # "num_augmentations":num_augmentations,
    # "num_techniques":num_techniques_per_sub_policy,
    # "num_sub_policies":num_sub_policies_per_policy,
    # "sub_policy_size":(num_augmentations + 2)* num_techniques_per_sub_policy,

    neighbour_move_functions = [NeighbourByTechnique
                                ,NeighbourByProbability
                                ,NeighbourByMagnitude
                                ]
    
    neighbour_move_value_ranges = [ None
                                    ,None
                                    ,None
                                     ]



    neighbours = []
    for sub_policy_i in range(species_attributes["num_sub_policies"]):
        for technique_i in range(species_attributes["num_techniques"]):
            current_technique, _, _, _ = FetchTechniqueFromLocalIndex(chromosome,technique_i, sub_policy_i,species_attributes)
            current_augmentation_local_i = current_technique[0].index(1.0)
            
            global_probability_i = MapLocalProbabilityIndexsToGlobalInChromosome(technique_i, sub_policy_i, species_attributes)
                
            global_magnitude_i = MapLocalMagnitudeIndexsToGlobalInChromosome(technique_i, sub_policy_i, species_attributes)
            
            #set possible values of technique for current technique
            neighbour_move_value_ranges[0] = list(range(species_attributes["num_augmentations"]-1))
            neighbour_move_value_ranges[0].remove(current_augmentation_local_i)
            
            #set possible values of probability for current technique
            neighbour_move_value_ranges[1] = []
            current_probability = chromosome[global_probability_i]
            if(current_probability >= 0.1):
                neighbour_move_value_ranges[1].append(current_probability-0.1)

            if(current_probability <= 0.9):
                neighbour_move_value_ranges[1].append(current_probability+0.1)
            
            #set possible values of magnitude for current technique
            neighbour_move_value_ranges[2] = []

            technique_name = species_attributes["augmentation_list"][current_augmentation_local_i]
            if technique_name in species_attributes["constant_magnitude_augmentations"]:
                neighbour_move_value_ranges[2].append(1.0)
            else:
                current_magnitude = chromosome[global_magnitude_i]
                if(current_magnitude >= 0.1):
                    neighbour_move_value_ranges[2].append(current_magnitude-0.1)

                if(current_magnitude <= 0.9):
                    neighbour_move_value_ranges[2].append(current_magnitude+0.1)
                


            for neighbour_move_func_i in range(len(neighbour_move_functions)):
                neighbour_move_func = neighbour_move_functions[neighbour_move_func_i]

                for neighbour_move_value in neighbour_move_value_ranges[neighbour_move_func_i]:
                    neighbours.append( CreateNeighbour(chromosome,species_attributes, sub_policy_i, technique_i, neighbour_move_func, neighbour_move_value) )

    return neighbours


def CreateNeighbour(chromosome,species_attributes, change_sub_policy_i = -1, change_technique_i=-1, neighbourhood_move_func = None , new_val =-1):
    if(change_sub_policy_i == -1):
        change_sub_policy_i = random.randint(0,species_attributes["num_sub_policies"]-1)
    
    if(change_technique_i == -1):
        change_technique_i = random.randint(0,species_attributes["num_techniques"]-1)
    
    if(neighbourhood_move_func is None):
        neighbourhood_move_func = random.choice([NeighbourByTechnique,NeighbourByMagnitude,NeighbourByProbability]) 

    return neighbourhood_move_func(chromosome,change_sub_policy_i,change_technique_i,species_attributes,new_val)
    


def NeighbourByTechnique(chromosome, s_i, t_i,species_attributes,new_val=-1):
    current_technique, t_global_i, _, _ = FetchTechniqueFromLocalIndex(chromosome,t_i, s_i,species_attributes)
    current_augmentation_local_i = current_technique[0].index(1.0)

    if(new_val != -1):
        new_augmentation_local_i = new_val
    else:
        new_augmentation_local_i = current_augmentation_local_i
        while(new_augmentation_local_i == current_augmentation_local_i):
            new_augmentation_local_i = random.randint(0,species_attributes["num_augmentations"]-1)

    new_chomosome = np.copy(chromosome)  
    new_chomosome[t_global_i+current_augmentation_local_i] = 0.0
    new_chomosome[t_global_i+new_augmentation_local_i] = 1.0

    return new_chomosome


def NeighbourByProbability(chromosome, s_i, t_i,species_attributes,new_val=-1):
    global_probability_i = MapLocalProbabilityIndexsToGlobalInChromosome(t_i, s_i, species_attributes)

    if(new_val != -1):
        new_probability = new_val
    else:
        probability_possibilities = []
        current_probability = chromosome[global_probability_i]
        if(current_probability >= 0.1):
            probability_possibilities.append(current_probability-0.1)

        if(current_probability <= 0.9):
            probability_possibilities.append(current_probability+0.1)

        new_probability = random.choice(probability_possibilities)

    new_chomosome = np.copy(chromosome)  
    
    new_chomosome[global_probability_i] = new_probability

    return new_chomosome


def NeighbourByMagnitude(chromosome, s_i, t_i,species_attributes,new_val=-1):
    global_magnitude_i = MapLocalMagnitudeIndexsToGlobalInChromosome(t_i, s_i, species_attributes)

    if(new_val != -1):
        new_magnitude = new_val
    else:
        magnitude_possibilities = []
        current_magnitude = chromosome[global_magnitude_i]
        if(current_magnitude >= 0.1):
            magnitude_possibilities.append(current_magnitude-0.1)

        if(current_magnitude <= 0.8):
            magnitude_possibilities.append(current_magnitude+0.1)
        
        new_magnitude = random.choice(magnitude_possibilities)

    new_chomosome = np.copy(chromosome)  
    
    new_chomosome[global_magnitude_i] = new_magnitude

    return new_chomosome


def CreateNeighboursFromChromosome(num_neighbours, chromosome, species_attributes ):
    neighbours_list = []

    for n_i in range(num_neighbours):
        neighbour = CreateNeighbour(chromosome,species_attributes)
        neighbours_list.append(neighbour)
    
    return neighbours_list





###MAPPING and FORMATTING FUNCTIONS
def MapLocalAugmentationIndexsToGlobalInChromosome(augmentation_i, technique_i, sub_policy_i, species_attributes):
    #jump to right sub policy + jump to the right technique vector + find the augmentation bit
    return (species_attributes["sub_policy_size"] * sub_policy_i) + (species_attributes["num_augmentations"]* technique_i) + augmentation_i

def MapLocalProbabilityIndexsToGlobalInChromosome(technique_i, sub_policy_i, species_attributes):
    #jump to right sub policy + jump over the technique vectors + jump to the right probability
    return  (species_attributes["sub_policy_size"] * sub_policy_i) + species_attributes["total_sp_one_hot_elements"] + technique_i 

def MapLocalMagnitudeIndexsToGlobalInChromosome(technique_i, sub_policy_i, species_attributes):
    #jump to right sub policy + jump over the technique vectors + jump over the probability floats + jump to the right magnitude
    return  (species_attributes["sub_policy_size"] * sub_policy_i) + species_attributes["total_sp_one_hot_elements"] + species_attributes["num_techniques"] + technique_i 


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
    augmentation_i = list(technique[0]).index(1.0)

    return (augmentations[augmentation_i], technique[1], int(technique[2]*10))


def PolicyChromosomeToSubPoliciesList(chromosome, augmentations, species_attributes):
    sub_policies = []
    for s_i in range(species_attributes["num_sub_policies"]):
        selected_augmentations = []
        techniques = []
        for t_i in range(species_attributes["num_techniques"]):
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
    

def PolicyJsonToChromosome(policy_json,augmentation_list,species_attributes):
    loaded_chromosome = BuildBlankPolicyChromosome(species_attributes)

    policy_dict = policy_json["policy"]

    num_subpolicies_in_json = len(policy_dict["policy"])
    num_techniques_in_json = len(policy_dict["policy"][0])

    for sub_policy_i in range(num_subpolicies_in_json):
        sub_policy_list = policy_dict["policy"][sub_policy_i]
        for technique_i in range(num_techniques_in_json):
            augmentation_i = augmentation_list.index(sub_policy_list[technique_i][0])
            global_augmentation_i = MapLocalAugmentationIndexsToGlobalInChromosome(augmentation_i, technique_i, sub_policy_i, species_attributes)
            loaded_chromosome[global_augmentation_i] = 1.0

            global_probability_i = MapLocalProbabilityIndexsToGlobalInChromosome(technique_i, sub_policy_i, species_attributes)
            loaded_chromosome[global_probability_i] = float(sub_policy_list[technique_i][1])

            global_magnitude_i = MapLocalMagnitudeIndexsToGlobalInChromosome(technique_i, sub_policy_i, species_attributes)
            loaded_chromosome[global_magnitude_i] = float(sub_policy_list[technique_i][2])/10
    
    return loaded_chromosome


##GENERATE CHOMOSOME FUNCTIONS
def BuildBlankPolicyChromosome(species_attributes):
    # format: [ [one hot encoded technique]*num_technique + [probabilities]*num_technique + [magnitude]*num_technique ] *num_sub_policies
    # EG:   [1000000 0100000  0.1 0.6  0.1 0.6   1000000 0100000  0.1 0.6  0.1 0.6   1000000 0100000  0.1 0.6  0.1 0.6] (one hot encodings are also floats but represented as ints for readability)
    # EG:   [T11 T12          P11 P12  M11 M12   T21     T22      P21 P22  M21 M22   T31     T32      P31 P32  M31 M32]
    # EG:   [|            ^Sub Policy 1^     |   |                ^S2^            |  |               ^S3^            |]
    return [0.0] * (species_attributes["sub_policy_size"] * species_attributes["num_sub_policies"])   
    

def CreateRandomChromosome(species_attributes):
    chromosome = BuildBlankPolicyChromosome(species_attributes)
    for s_i in range(species_attributes["num_sub_policies"]):
        for t_i in range(species_attributes["num_techniques"]):
            augmentation_i = random.randint(0,species_attributes["num_augmentations"]-1)
            chromosome[ MapLocalAugmentationIndexsToGlobalInChromosome(augmentation_i, t_i, s_i, species_attributes) ] = 1.0

            magnitude = random.randint(0,10)/10.0
            chromosome[MapLocalMagnitudeIndexsToGlobalInChromosome(t_i, s_i, species_attributes)] = magnitude
            
            probability = random.randint(0,9)/10.0
            chromosome[MapLocalProbabilityIndexsToGlobalInChromosome(t_i, s_i, species_attributes)] = probability
            
    return chromosome


def GenerateStartingPoliciesFromSplittingPaperPolicy(population_size,augmentation_list,species_attributes):
    #Splits the 25 sub_policies of the paper policy in to policies of 5 sub-policies and generates neighbours of those
    #requires policies to be 5 sub-policies of 2 techniques

    paper_policy_path = "autoaugment_paper_cifar10.json"

    paper_policy_json = None

    with open(paper_policy_path, "r") as f:
        policy_string = f.read()
        paper_policy_json = json.loads(policy_string)

    paper_chromosome = PolicyJsonToChromosome(paper_policy_json,augmentation_list,species_attributes)

    policies = [paper_chromosome]


    policies += CreateNeighboursFromChromosome(population_size-1, paper_chromosome, species_attributes)

    return policies


def GenerateStartingPoliciesFromPaperPolicy(population_size,augmentation_list,species_attributes):
    #requires policies to be 25 sub-policies of 2 techniques

    paper_policy_path = "autoaugment_paper_cifar10.json"

    paper_policy_json = None

    with open(paper_policy_path, "r") as f:
        policy_string = f.read()
        paper_policy_json = json.loads(policy_string)

    paper_chromosome = PolicyJsonToChromosome(paper_policy_json,augmentation_list,species_attributes)

    policies = [paper_chromosome]


    policies += CreateNeighboursFromChromosome(population_size-1, paper_chromosome, species_attributes)

    return policies


def LoadGenerationByEvolutionStep(step,experiment_attributes):
    policies = []

    for chromosome_i in range(experiment_attributes["species_attributes"]["population_size"]):
        chromosome_id = format(step, '05d') + "_" + format(chromosome_i, '05d')

        policy_id = experiment_attributes["experiment_id"]+"_"+str(chromosome_id)
    
        policy_path = os.path.join("policies",policy_id+".json")

        policy_json = None

        with open(policy_path, "r") as f:
            policy_string = f.read()
            policy_json = json.loads(policy_string)

        loaded_chromosome = PolicyJsonToChromosome(policy_json,experiment_attributes["augmentation_list"],experiment_attributes["species_attributes"] )

        policies.append(loaded_chromosome)
    
    return policies


###EVOLUTION STAGE FUNCTIONS
def EvaluatePopulation(population,fitness_function, augmentations, experiment_attributes, step):
    population_fitness = []

    policies = []
    for chromosome_i in range(len(population)):
        chromosome_id = format(step, '05d') + "_" + format(chromosome_i, '05d')
        chromosome = population[chromosome_i]

        policies.append( (chromosome, StorePoliciesAsJsons(chromosome, augmentations, experiment_attributes, chromosome_id)) )
    
    population_fitness, best_policy_id, best_accuracy  = experiment_attributes["population_evaluation_function"](fitness_function, policies, augmentations, experiment_attributes)
    # for policy in range(len(policies)):
    #     population_fitness.append( (policy[0], fitness_function(policies[0],augmentations, experiment_attributes, policy[1])) )

    return population_fitness, best_policy_id, best_accuracy 


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
    num_elite = int(math.ceil((species_attributes["population_size"]*elitism_ratio)/2.0)*2)
    
    next_generation += [x[0] for x in population_fitness[:num_elite]]
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
def CreateSpeciesAttributeDict(population_size, num_augmentations, num_techniques_per_sub_policy, num_sub_policies_per_policy):
    return {
        "population_size": population_size,
        "num_augmentations":num_augmentations,
        "num_techniques":num_techniques_per_sub_policy,
        "num_sub_policies":num_sub_policies_per_policy,
        "sub_policy_size":(num_augmentations + 2)* num_techniques_per_sub_policy,
        "total_sp_one_hot_elements":num_augmentations * num_techniques_per_sub_policy
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

    keys = ["step","max_fitness","min_fitness","average_fitness","best_fitness_so_far","best_policy_so_far"]
    for k in keys:
        generation_string += str(generation_stats[k]) + ","
    
    generation_string = generation_string[:-1]+"\n"
    with open(experiment_id+".csv","a") as f:
        f.write(generation_string)


def LocalCleanDirectoriesAndStoreCurrentGen(policy_ids):
    previous_generation_path = "previous_generation"

    if(not os.path.exists(previous_generation_path)):
        os.mkdir(previous_generation_path)

    #clean previous_generation folders
    CleanLocalPreviousGeneration(previous_generation_path)
    
    #copy current generations to previous_generation folders
    CopyLocalCurrentGenerationToPreviousGenerationFolder(policy_ids,previous_generation_path)
    
    #clean main directories
    CleanLocalCurrentGeneration(policy_ids)


def CleanLocalPreviousGeneration(previous_generation_path):
    previous_checkpoints_path = os.path.join(previous_generation_path, "checkpoints")
    if(not os.path.exists(previous_checkpoints_path)):
        os.mkdir(previous_checkpoints_path)

    checkpoints = os.listdir(previous_checkpoints_path)
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(previous_checkpoints_path,checkpoint)
        shutil.rmtree(checkpoint_path)


    previous_policies_path = os.path.join(previous_generation_path, "policies")
    if(not os.path.exists(previous_policies_path)):
        os.mkdir(previous_policies_path)

    policies = os.listdir(previous_policies_path)
    for policy in policies:
        policy_path = os.path.join(previous_policies_path,policy)
        os.remove(policy_path)


def CopyLocalCurrentGenerationToPreviousGenerationFolder(policy_ids,previous_generation_path):
    checkpoints_dir = "checkpoints"
    policies_dir = "policies"

    previous_checkpoints_path = os.path.join(previous_generation_path, "checkpoints")
    previous_policies_path = os.path.join(previous_generation_path, "policies")
    
    for policy_id in policy_ids:
        checkpoint_path = os.path.join(checkpoints_dir,"checkpoints_"+policy_id)
        policy_path = os.path.join(policies_dir,policy_id+".json")

        checkpoint_output_path = os.path.join(previous_checkpoints_path,"checkpoints_"+policy_id)
        policy_output_path = os.path.join(previous_policies_path,policy_id+".json")

        shutil.copytree(checkpoint_path,checkpoint_output_path)
        shutil.copy(policy_path,policy_output_path)


def CleanLocalCurrentGeneration(policy_ids):
    checkpoints_dir = "checkpoints"
    policies_dir = "policies"

    for policy_id in policy_ids:
        checkpoint_path = os.path.join(checkpoints_dir,"checkpoints_"+policy_id)
        policy_path = os.path.join(policies_dir,policy_id+".json")

        shutil.rmtree(checkpoint_path)
        os.remove(policy_path)


        
if(__name__ == "__main__"):
    train_remote = True
    autoaugment_based_population = False
    load_from_step = 5
    
    data_path = "/media/harborned/ShutUpN/datasets/cifar/cifar-10-batches-py"
    if(len(sys.argv) > 1):
        data_path = sys.argv[1]
    experiment_attributes = {
        "experiment_id":"AutoAugmentBasedPopulation_exp_0002_120e_10p_25-2"
        ,"num_epochs":120
        ,"data_path":data_path
        ,"dataset":"cifar10"
        ,"model_name":"wrn"
        ,"use_cpu":0
        ,"clean_directories":True
        ,"num_steps": 10
       
    }

    if(train_remote):
        print("Training Remotely")
        experiment_attributes["population_evaluation_function"] = ArccaParallel
    else:
        print("Training Locally")
        experiment_attributes["population_evaluation_function"] = LocalSequential

    augmentation_list = list(augmentation_transforms.TRANSFORM_NAMES)
    augmentation_list = list(augmentation_transforms.FILTERED_TRANSFORM_NAMES)
	
    experiment_attributes["augmentation_list"] = augmentation_list
    constant_magnitude_augmentations = list(augmentation_transforms.IGNORES_MAGNITUDE_NAMES)
    experiment_attributes["constant_magnitude_augmentations"] = constant_magnitude_augmentations
    
    print("")
    print("number of augmentations: ", len(augmentation_list))
    print("")
    num_techniques_per_sub_policy = 2
    num_sub_policies_per_policy = 25
    
    population_size = 10

    prob_crossover = 0.01
    prob_technique_mutate = 0.05
    prob_magnitude_mutate = 0.05
    prob_probability_mutate = 0.05


    num_evolution_steps = experiment_attributes["num_steps"]

    fitness_function = TrainWithPolicyFitness
    

    evolution_probabilities = CreateProbabilitiesDict(prob_crossover,prob_technique_mutate,prob_probability_mutate,prob_magnitude_mutate)

    species_attributes = CreateSpeciesAttributeDict(population_size,len(augmentation_list), num_techniques_per_sub_policy, num_sub_policies_per_policy)
    
    species_attributes["augmentation_list"] = augmentation_list
    species_attributes["constant_magnitude_augmentations"] = constant_magnitude_augmentations

    experiment_attributes["species_attributes"] = species_attributes

    experiment_attributes["local_ga_directory"] = "/media/harborned/ShutUpN/repos/final_year_project/genetic_augment"
    experiment_attributes["remote_ga_directory"] = "/home/c.c0919382/fyp_scw1427/genetic_augment"

    population = []
    
    starting_step = 0 
    if(autoaugment_based_population):
        population = GenerateStartingPoliciesFromPaperPolicy(population_size,augmentation_list,species_attributes)
    else:
        if(load_from_step == -1):
            for p_i in range(population_size):
                population.append(CreateRandomChromosome(species_attributes))
        else:
            population = LoadGenerationByEvolutionStep(load_from_step,experiment_attributes)    
            starting_step = load_from_step

    best_global_accuracy = 0
    best_global_policy_id = None
    for step in range(starting_step,num_evolution_steps):
        print("____")
        print("Starting evolution step: " +str(step))
        print("")

        population_fitness, best_policy_id, best_accuracy = EvaluatePopulation(population,fitness_function, augmentation_list, experiment_attributes, step)
        if(best_accuracy > best_global_accuracy):
            best_global_policy_id = best_policy_id
            best_global_accuracy = best_accuracy
      
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
                ,"best_fitness_so_far":best_global_accuracy
                ,"best_policy_so_far":best_global_policy_id
            }

        LogGeneration(experiment_attributes["experiment_id"],generation_stats)

        if(step % 1==0):
            print("step:",step)
            print("Max Fitness:", generation_stats["max_fitness"])
            print("Min Fitness:", generation_stats["min_fitness"] )
            print("Average Fitness:", generation_stats["average_fitness"] )
            print("Best Fitness So Far:", best_global_accuracy)
            print("Best Policy Id So Far:", best_global_policy_id)
            
            # for p in population_fitness:
            #     print(p)
            print("-----")
            print("")

            


        population = MoveToNextGeneration(population_fitness, evolution_probabilities, species_attributes, elitism_ratio=0.1)
    print("Best Policy Found:")
    print(best_global_policy_id)
    print(best_global_accuracy)

