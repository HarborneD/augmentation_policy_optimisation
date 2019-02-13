from trainer import augmentation_transforms

import sys

# Import modules
import numpy as np

# Import PySwarms
import pyswarms as ps






#ATTRIBUTE DICTIONARY FUNCTIONS
def CreateSpeciesAttributeDict(population_size, num_augmentations, num_technqiues_per_sub_policy, num_sub_policies_per_policy):
    return {
        "population_size": population_size,
        "num_augmentations":num_augmentations,
        "num_technqiues":num_technqiues_per_sub_policy,
        "num_sub_policies":num_sub_policies_per_policy,
        "sub_policy_size":(num_augmentations + 2)* num_technqiues_per_sub_policy,
        "total_sp_one_hot_elements":num_augmentations * num_technqiues_per_sub_policy,
        "bounds_size": 3 * num_augmentations
    }


def CreateProbabilitiesDict(prob_crossover,prob_technique_mutate,prob_probability_mutate,prob_magnitude_mutate):
    return {
        "prob_crossover": prob_crossover,
        "prob_technique_mutate": prob_technique_mutate,
        "prob_probability_mutate": prob_probability_mutate, 
        "prob_magnitude_mutate": prob_magnitude_mutate 
    }



##GENERATE CHOMOSOME FUNCTIONS
def BuildBounds(species_attributes):
    min_bound = [0.0] * species_attributes["num_augmentations"] +[0.0] * species_attributes["num_augmentations"] + [0.1] * species_attributes["num_augmentations"]    
    max_bound = [1.0] * (3 * species_attributes["num_augmentations"])
    return min_bound,max_bound 

if(__name__ == "__main__"):
    data_path = "/media/harborned/ShutUpN/datasets/cifar/cifar-10-batches-py"
    if(len(sys.argv) > 1):
        data_path = sys.argv[1]
    experiment_attributes = {
        "experiment_id":"test_exp_0001_20e_10p_5-2"
        ,"num_epochs":20
        ,"data_path":data_path
        ,"dataset":"cifar10"
        ,"model_name":"wrn"
        ,"use_cpu":0
        #,"population_evaluation_function": LocalSequential
    }

def MapParticleToPolicy(particle):
    #create blank sub policy list

    #check the order vector
    
    #order techniques by order vector 

    #place probabilities and magnitudes in sub policiy
    pass
    



    augmentation_list = list(augmentation_transforms.TRANSFORM_NAMES)
    print("")
    print("number of augmentations: ", len(augmentation_list))
    print("")
    num_technqiues_per_sub_policy = len(augmentation_list)
    num_sub_policies_per_policy = 1
    
    population_size = 10

    prob_crossover = 0.001
    prob_technique_mutate = 0.001
    prob_magnitude_mutate = 0.001
    prob_probability_mutate = 0.001


    num_evolution_steps = 10000

    #fitness_function = TrainWithPolicyFitness
    
    def TestFunc(x):
        print(x.shape)
        f = np.array([sum(x_i) for x_i in x])
        print("f shape:",f.shape)
        return f 

    def rosenbrock_with_args(x, a, b, c=0):
        print("x shape:", x.shape)
        f = (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2 + c
        print("f shape:",f.shape)
        return f

        
    fitness_function = TestFunc

    evolution_probabilities = CreateProbabilitiesDict(prob_crossover,prob_technique_mutate,prob_probability_mutate,prob_magnitude_mutate)

    species_attributes = CreateSpeciesAttributeDict(population_size,len(augmentation_list), num_technqiues_per_sub_policy, num_sub_policies_per_policy)
    
    experiment_attributes["species_attributes"] = species_attributes


    min_bound,max_bound = BuildBounds(species_attributes)
    bounds = (min_bound, max_bound)

    #fitness_function = rosenbrock_with_args
    # max_bound = 5.12 * np.ones(2)
    # min_bound = - max_bound
    # bounds = (min_bound, max_bound)

    # Initialize swarm
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Call instance of PSO with bounds argument
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=len(min_bound), options=options, bounds=bounds)

    # Perform optimization
    cost, pos = optimizer.optimize(fitness_function, iters=1000)#,a=1, b=100, c=0)