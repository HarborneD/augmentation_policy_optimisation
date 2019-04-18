from evaluator import augmentation_transforms

import sys
import os
import json

import time

import math

# Import modules
import numpy as np

# Import PySwarms
import pyswarms as ps

import threading

# from ArccaGAFunctions import RemoteGATool
from GeneticAugment import ArccaParallel

class AtomicInteger():
    def __init__(self, value=0):
        self._value = value
        self._lock = threading.Lock()

    def inc(self):
        with self._lock:
            self._value += 1
            return self._value

    def dec(self):
        with self._lock:
            self._value -= 1
            return self._value

    def update(self,new_val):
        with self._lock:
            old_val = self._value
            self._value = new_val
            return self._value, old_val


    def update_if_threshold(self,new_val,threshold):
        with self._lock:
            old_val = self._value
            if(new_val - old_val >= threshold):
                self._value = new_val
                return True, self._value, old_val
            return False, self._value, old_val 


    @property
    def value(self):
        with self._lock:
            return self._value

    @value.setter
    def value(self, v):
        with self._lock:
            self._value = v
            return self._value


def StorePolicyDictAsJson(policy_id, policy_dict, policy_directory="policies"):
    policy_json = json.dumps(policy_dict,indent=4)
    
    policy_json_path = os.path.join(policy_directory, policy_id+".json")
    with open(policy_json_path, "w") as f:
        f.write(policy_json)


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



##GENERATE PARTICLE FUNCTIONS
def BuildBounds(species_attributes):
    min_bound = [0.0] * species_attributes["num_augmentations"] +[0.0] * species_attributes["num_augmentations"] + [0.1] * species_attributes["num_augmentations"]    
    max_bound = [1.0] * (3 * species_attributes["num_augmentations"])
    return min_bound,max_bound 



def MapParticleToPolicy(particle,augmentation_list,policy_id=""):
    #create blank sub policy list
    policy_dict = {
        "policy": {
            "policy": [ #policy
                [       #sub-policy
                      #[] * num_techniques -  technique 
                ]
            ], 
            "id": str(policy_id)
        }
    }

    #check the order vector
    order_vector = particle[:species_attributes["num_augmentations"]]
    
    order_tuples_list = [(order_vector[i],i) for i in range(len(order_vector))]
    
    #order techniques by order vector 
    order_tuples_list.sort()
    

    #place probabilities and magnitudes in sub policiy
    probability_vector = particle[species_attributes["num_augmentations"]:2*species_attributes["num_augmentations"]]
    magnitude_vector = particle[2*species_attributes["num_augmentations"]:3*species_attributes["num_augmentations"]]
    
    
    for order_tuple in order_tuples_list:
        augmentation_index = order_tuple[1]
        
        augmentation_name = augmentation_list[augmentation_index]
        augmentation_probability = round(probability_vector[augmentation_index],1)
        augmentation_magnitude = round(10*magnitude_vector[augmentation_index],0)

        policy_dict["policy"]["policy"][0].append([augmentation_name, augmentation_probability, augmentation_magnitude])
    
    
    return policy_dict

def ArccaEvaluateParticles(particles, augmentations, experiment_attributes, policy_id_int):
    policy_ids = []
    policies = []
    
    #store all particles as policies
    for particle in particles:
        chromosome_id = format(policy_id_int.inc(), '05d')
        policy_id = experiment_attributes["experiment_id"]+"_"+str(chromosome_id)

        policy_dict = MapParticleToPolicy(particle,augmentation_list,policy_id=policy_id)

        StorePolicyDictAsJson(policy_id, policy_dict, policy_directory="policies")
        policy_ids.append(policy_id)
        policies.append((particle,policy_id))
    
    population_fitness, best_policy_id, best_accuracy = ArccaParallel(None, policies, augmentations, experiment_attributes, repeats_per_policy=experiment_attributes["repeats_per_policy"])

    results_dict = dict([(p[2],1 - p[1]) for p in population_fitness])

    fitness_values = [results_dict[id] for id in policy_ids]
    
    return np.array(fitness_values)

# def ArccaEvaluateParticle(particle, augmentation_list, experiment_attributes, policy_id_int, job_statuses, job_last_update):
#         remote_tool = RemoteGATool(experiment_attributes["local_ga_directory"],experiment_attributes["remote_ga_directory"])

#         chromosome_id = format(policy_id_int.inc(), '05d')
#         policy_id = experiment_attributes["experiment_id"]+"_"+str(chromosome_id)

#         policy_dict = MapParticleToPolicy(particle,augmentation_list,policy_id=policy_id)

#         #save policy file
#         StorePolicyDictAsJson(policy_id, policy_dict, policy_directory="policies")
    
#         #send policy file to arcca
#         remote_tool.SendPolicyFile(policy_id)

#         cumulative_results = {}
#         repeats_per_policy = experiment_attributes["repeats_per_policy"]
#         for repeat_i in range(repeats_per_policy):
#             #submit job
#             job_id, was_error = remote_tool.StartRemoteChromosomeTrain(policy_id, experiment_attributes["num_epochs"], experiment_attributes["data_path"], dataset="cifar10", model_name="wrn", use_cpu=0, num_train_images=experiment_attributes["num_train_images"])
            
#             if was_error:
#                 print("error posting job: "+str(job_id))
            
#             #wait for job to complete
#             job_status = ""
#             wait_for_status_check_time = 2
#             while job_status != "COMPLETED":
#                 time.sleep(2)

#                 update, new_val, old_val = job_last_update.update_if_threshold(int(time.time()), wait_for_status_check_time )

#                 if(update):
#                     job_statuses = remote_tool.arcca_tool.CheckJobsStatuses(start_time="2019-03-01")

#                 if(job_id in job_statuses):
#                    job_status = job_statuses[job_id]
#                 else:
#                     print("job not in job statuses: "+str(job_id)) 

            
#             #fetch results
#             result = remote_tool.GetPolicyResults(policy_id)

#             cumulative_results[policy_id] += result["test_accuracy"]

#             if(repeat_i != (repeats_per_policy-1)):
#                 remote_tool.CleanCheckpoints([policy_id])
        
#         #aggregate trials and return 
#         return cumulative_results[policy_id] / float(repeats_per_policy)


        
        
    

if(__name__ == "__main__"):
    data_path = "/media/harborned/ShutUpN/datasets/cifar/cifar-10-batches-py"
    if(len(sys.argv) > 1):
        data_path = sys.argv[1]
    experiment_attributes = {
        "num_epochs":20
        ,"repeats_per_policy":3
        ,"num_train_images":4000
        ,"data_path":data_path
        ,"dataset":"cifar10"
        ,"model_name":"wrn"
        ,"use_cpu":0
        ,"clean_directories":True
        ,"num_particles": 10
        ,"num_steps": 100
        
        #,"population_evaluation_function": LocalSequential
    }
    experiment_attributes["experiment_id"] = "pso_exp_0002_"+str(experiment_attributes["num_epochs"])+"e_"+str(experiment_attributes["num_particles"])+"p_1-20"
        

    experiment_attributes["local_ga_directory"] = "/media/harborned/ShutUpN/repos/final_year_project/genetic_augment"
    experiment_attributes["remote_ga_directory"] = "/home/c.c0919382/fyp_scw1427/genetic_augment"


    augmentation_list = list(augmentation_transforms.TRANSFORM_NAMES)
    augmentation_list = list(augmentation_transforms.FILTERED_TRANSFORM_NAMES)
	
    augmentation_list.sort()
    print("")
    print("number of augmentations: ", len(augmentation_list))
    print("")
    num_technqiues_per_sub_policy = len(augmentation_list)
    num_sub_policies_per_policy = 1

    num_particles = experiment_attributes["num_particles"]
    num_steps =  experiment_attributes["num_steps"]

    #fitness_function = TrainWithPolicyFitness

    # def TestFunc(x):
    #     f = np.array([sum(x_i) for x_i in x])
    #     return f 

    # def rosenbrock_with_args(x, a, b, c=0):
    #     f = (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2 + c
    #     return f


    policy_id = AtomicInteger()

    fitness_function = ArccaEvaluateParticles

    species_attributes = CreateSpeciesAttributeDict(num_particles,len(augmentation_list), num_technqiues_per_sub_policy, num_sub_policies_per_policy)

    experiment_attributes["species_attributes"] = species_attributes


    min_bound,max_bound = BuildBounds(species_attributes)
    bounds = (min_bound, max_bound)

    # #fitness_function = rosenbrock_with_args
    # # max_bound = 5.12 * np.ones(2)
    # # min_bound = - max_bound
    # # bounds = (min_bound, max_bound)

    # Initialize swarm
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9}

    # Call instance of PSO with bounds argument
    optimizer = ps.single.GlobalBestPSO(n_particles=num_particles, dimensions=len(min_bound), options=options, bounds=bounds)

    # Perform optimization
    job_statuses = {}
    job_last_update = AtomicInteger()

    cost, pos = optimizer.optimize(fitness_function, iters=num_steps, augmentations=augmentation_list, experiment_attributes=experiment_attributes, policy_id_int = policy_id)

    print(cost)
    print(pos)

    # print("augmentation_list")
    # for aug in augmentation_list:
    #     print(aug)
    
    # test_particle = [0] * (3*len(augmentation_list))
    # for i in range(len(augmentation_list)):
    #     test_particle[i] = (len(augmentation_list) - i) / 20.0
    #     test_particle[i + len(augmentation_list)] = (i) / 20.0
    #     test_particle[i + 2*len(augmentation_list)] = (len(augmentation_list) - i) / 20.0
    

    # print(test_particle)

    # policy_dict = MapParticleToPolicy(test_particle, augmentation_list, "test_00064")

    # for sub in policy_dict["policy"]["policy"]:
    #     for technique in sub:
    #         print(technique)
