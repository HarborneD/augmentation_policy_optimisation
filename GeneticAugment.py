from trainer import augmentation_transforms

from StochasticUniversalSampler import StochasticUniversalSampler

import numpy as np
import random
import math 



def TestFitnessWithPolicy(chromosome, augmentations, species_attributes):
    policy = PolicyChromosomeToSubPoliciesList(chromosome, augmentations, species_attributes)
    # print("Policy:")
    # for sp in policy:
    #     print(sp)

    # print("run training")

    return TestFitness2(chromosome, augmentations, species_attributes)


def CreateJob(policy, model, epochs, dataset):
    pass


def TestFitness(chromosome, augmentations, species_attributes):
    def EvaluateTechnique(technique):
        current_augmentation_local_i = technique[0].index(1.0)

        return technique[1] * (current_augmentation_local_i +1)

    policy_val = 0
    for s_i in range(species_attributes["num_sub_policies"]):
        techniques = []
        for t_i in range(species_attributes["num_technqiues"]):
            tech, _, _ = FetchTechniqueFromLocalIndex(chromosome,t_i,s_i,species_attributes)
            techniques.append( tech )
        
        sub_policy_val = 10 - abs(2 - EvaluateTechnique(techniques[0]) - EvaluateTechnique(techniques[1]))    
        policy_val += sub_policy_val

    return policy_val


def TestFitness2(chromosome, augmentations, species_attributes):
    ### fitness = sum( foreach subpolicy: foreach technique: technique_index )
    ### max = num_sub_policies * 2 * (num_augmentations-1)
    def EvaluateTechnique(technique):
        current_augmentation_local_i = technique[0].index(1.0)

        return current_augmentation_local_i
    
    policy_val = 0
    for s_i in range(species_attributes["num_sub_policies"]):
        techniques = []
        for t_i in range(species_attributes["num_technqiues"]):
            tech, _, _ = FetchTechniqueFromLocalIndex(chromosome,t_i,s_i,species_attributes)
            techniques.append( tech )
        
        sub_policy_val = EvaluateTechnique(techniques[0]) + EvaluateTechnique(techniques[1])   
        policy_val += sub_policy_val

    return policy_val

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
                
                current_technique, t_global_i, t_magnitude_global_i = FetchTechniqueFromLocalIndex(chromosome,t_i, s_i,species_attributes)
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

                if(current_magnitude <= 0.9):
                    magnitude_possibilities.append(current_magnitude+0.1)
                
                new_magnitude = random.choice(magnitude_possibilities)

                chromosome[global_magnitude_i] = new_magnitude
    
    return chromosome




def MapLocalAugmentationIndexsToGlobalInChromosome(augmentation_i, technique_i, sub_policy_i, species_attributes):
    return augmentation_i + (species_attributes["num_augmentations"]* technique_i) + (species_attributes["sub_policy_size"] * sub_policy_i)

def MapLocalMagnitudeIndexsToGlobalInChromosome(technique_i, sub_policy_i, species_attributes):
    return  species_attributes["total_sp_one_hot_elements"] + technique_i + (species_attributes["sub_policy_size"] * sub_policy_i)

def FetchTechniqueFromLocalIndex(chromosome, technique_i, sub_policy_i, species_attributes):
    augmentation_vector_start_i = MapLocalAugmentationIndexsToGlobalInChromosome(0, technique_i, sub_policy_i, species_attributes)

    magnitude_index = MapLocalMagnitudeIndexsToGlobalInChromosome(technique_i, sub_policy_i, species_attributes)
    # print(technique_i, sub_policy_i)
    # print(chromosome)
    # print(magnitude_index)
    # print(augmentation_vector_start_i)

    return (chromosome[augmentation_vector_start_i:augmentation_vector_start_i+species_attributes["num_augmentations"]] , chromosome[magnitude_index]), augmentation_vector_start_i, magnitude_index

def FetchSubPolicyFromIndex(chromosome, sub_policy_i, species_attributes):
    sp_global_start_index = MapLocalAugmentationIndexsToGlobalInChromosome(0, 0, sub_policy_i, species_attributes)

    return chromosome[sp_global_start_index:sp_global_start_index+species_attributes["sub_policy_size"]], sp_global_start_index


def TechniqueToAugmentationTuple(technique,augmentations):
    augmentation_i = technique[0].index(1.0)

    return (augmentations[augmentation_i], 1.0, technique[1])


def PolicyChromosomeToSubPoliciesList(chromosome, augmentations, species_attributes):
    sub_policies = []
    for s_i in range(species_attributes["num_sub_policies"]):
        selected_augmentations = []
        techniques = []
        for t_i in range(species_attributes["num_technqiues"]):
            tech, _, _ = FetchTechniqueFromLocalIndex(chromosome,t_i,s_i,species_attributes)
            techniques.append(tech[:])
        
        for technique in techniques:
            selected_augmentations.append(TechniqueToAugmentationTuple(technique,augmentations))
        sub_policies.append({"s_i":s_i, "augmentations":selected_augmentations[:]})    

    return sub_policies


def BuildBlankPolicyChromosome(species_attributes):
    # format: [ [one hot encoded technique]*num_technique + [magnitude]*num_technique ] *num_sub_policies
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
            
    return chromosome


def EvaluatePopulation(population,fitness_function, augmentations, species_attributes):
    population_fitness = []

    for chromosome in population:
        population_fitness.append( (chromosome, fitness_function(chromosome,augmentations, species_attributes)) )
    
    return population_fitness


def PerformCrossoverAndMutations(chromosome_1,chromosome_2, evolution_probabilities, species_attributes):
    #print("chromosome_1",chromosome_1)
    chromosome_1,chromosome_2 = CrossoverPolicy(chromosome_1,chromosome_2,evolution_probabilities["prob_crossover"],species_attributes)
    
    chromosome_1 = MutateTechnique(chromosome_1, evolution_probabilities["prob_technique_mutate"],species_attributes)
    chromosome_2 = MutateTechnique(chromosome_2, evolution_probabilities["prob_technique_mutate"],species_attributes)

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
    


def CreateSpeciesAttributeDict(population_size, num_augmentations, num_technqiues_per_sub_policy, num_sub_policies_per_policy):
    return {
        "population_size": population_size,
        "num_augmentations":num_augmentations,
        "num_technqiues":num_technqiues_per_sub_policy,
        "num_sub_policies":num_sub_policies_per_policy,
        "sub_policy_size":(num_augmentations + 1)* num_technqiues_per_sub_policy,
        "total_sp_one_hot_elements":num_augmentations * num_technqiues_per_sub_policy
    }


def CreateProbabilitiesDict(prob_crossover,prob_technique_mutate,prob_magnitude_mutate):
    return {
        "prob_crossover": prob_crossover,
        "prob_technique_mutate": prob_technique_mutate,
        "prob_magnitude_mutate": prob_magnitude_mutate 
    }


if(__name__ == "__main__"):
    augmentation_list = list(augmentation_transforms.TRANSFORM_NAMES)
    print("number of augmentations: ", len(augmentation_list))
    num_technqiues_per_sub_policy = 2
    num_sub_policies_per_policy = 5
    
    population_size = 10

    prob_crossover = 0.001
    prob_technique_mutate = 0.001
    prob_magnitude_mutate = 0.001

    num_evolution_steps = 10000

    fitness_function = TestFitnessWithPolicy
    

    evolution_probabilities = CreateProbabilitiesDict(prob_crossover,prob_technique_mutate,prob_magnitude_mutate)

    species_attributes = CreateSpeciesAttributeDict(population_size,len(augmentation_list), num_technqiues_per_sub_policy, num_sub_policies_per_policy)

    population = []
    for p_i in range(population_size):
        population.append(CreateRandomChromosome(species_attributes))
    

    # for p in population:
    #     print(p)

    # print("bbbbb")


    for step in range(num_evolution_steps):

        population_fitness = EvaluatePopulation(population,fitness_function, augmentation_list, species_attributes)
        
      
        # for p in population_fitness:
        #     print(p)

        # print("fffffff")
        # print("")

     

        fitness_vals = [x[1]for x in population_fitness]
        if(step % 500==0):
            print("step:",step)
            print("Max Fitness:", max(fitness_vals))
            print("Min Fitness:", min(fitness_vals))
            print("Average Fitness:", sum(fitness_vals)/float(len(fitness_vals)))
            # for p in population_fitness:
            #     print(p)
            print("-----")
            print("")
        population = MoveToNextGeneration(population_fitness, evolution_probabilities, species_attributes, elitism_ratio=0.05)


# for p in population:
#     print(p)

# print("fffff")


print(population[0])
print(fitness_function(population[0], augmentation_list, species_attributes))