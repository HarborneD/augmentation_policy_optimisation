import math
import numpy as np
#policies consist of N sub-policies
#sub-policies consist of Q technique configuration
#Technique configurations consist of an augmentation technique, a probability of application and a magnitude of application






#evolution strategy

#number of elite policies moved straight in to next
#num_elite = int(math.ceil((population_size*elitism_ratio)/2.0)*2)

#select using SUS an additional number of policies (population_size - num_elite)

#Evolve the selected:
#for each pair:
    #apply crossover
    #mutate technique
        #for each technique in all technique configurations, if prob < chance to mutuate, mutate to new technique
    #mutate probability
    #mutate magnitude


#if a pair contains one policy which is one magniture step away from optimal , prob of adding the optimal policy to next gen is:


def CalculateProbabilityOfOutcome(requirements, probabilities, population_attributes):
    num_crossovers_required = requirements["num_crossovers_required"]
    num_technique_mutation_required = requirements["num_technique_mutation_required"]
    num_magnitude_mutation_required = requirements["num_magnitude_mutation_required"]
    num_probability_mutation_required = requirements["num_probability_mutation_required"]

    prob_crossover = probabilities["prob_crossover"]
    prob_technique_mutation = probabilities["prob_technique_mutation"]
    prob_magnitude_mutation = probabilities["prob_magnitude_mutation"]
    prob_probability_mutation = probabilities["prob_probability_mutation"]

    num_sub_policies = population_attributes["num_sub_policies"]
    num_technique = population_attributes["num_technique"]

    population_size = population_attributes["population_size"]
    elitism_ratio = population_attributes["elitism_ratio"]

    probability_of_outcome = 1

    #prob crossover correctly 
    total_possible_crossover = 1
    prob_not_crossover = (1 - prob_crossover)
    prob_get_correct_crossover = 1 / float(num_sub_policies-1)

    prob_crossover_correctly = ( (prob_crossover*prob_get_correct_crossover)**num_crossovers_required) * (prob_not_crossover**(total_possible_crossover - num_crossovers_required) )

    #prob not mutate technique any technique
    total_possible_technique_mutation = num_sub_policies * num_technique
    prob_not_technique_mutation = (1 - prob_technique_mutation)
    prob_get_correct_technique = 1 / float(num_technique-1)
    prob_technique_mutate_correctly = ( (prob_technique_mutation*prob_get_correct_technique)**num_technique_mutation_required) * (prob_not_technique_mutation**(total_possible_technique_mutation - num_technique_mutation_required) )

    #prob not mutuate any probabilities
    total_possible_probability_mutation = num_sub_policies * num_technique
    prob_not_probability_mutation = (1 - prob_probability_mutation)
    prob_get_correct_probability = 0.5 #does nto factor in current value is 0.0 or 1.0
    prob_probability_mutate_correctly = ( (prob_probability_mutation*prob_get_correct_probability) **num_probability_mutation_required) * (prob_not_probability_mutation**(total_possible_probability_mutation - num_probability_mutation_required) )

    #prob mututate the correct magnitude
    total_possible_magnitude_mutation = num_sub_policies * num_technique
    prob_not_magnitude_mutation = (1 - prob_magnitude_mutation)
    prob_get_correct_magnitude = 0.5 #does nto factor in current value is 0.0 or 1.0

    prob_magnitude_mutate_correctly = ( (prob_magnitude_mutation*prob_get_correct_magnitude)**num_magnitude_mutation_required) * (prob_not_magnitude_mutation**(total_possible_magnitude_mutation - num_magnitude_mutation_required) )

    probability_of_outcome *= prob_crossover_correctly
    probability_of_outcome *= prob_technique_mutate_correctly
    probability_of_outcome *= prob_probability_mutate_correctly
    probability_of_outcome *= prob_magnitude_mutate_correctly

    return probability_of_outcome










if __name__ == "__main__":
        
    population_attributes = {
        "num_sub_policies": 2 #N
        ,"num_technique": 2 # Q
        ,"population_size": 10
        ,"elitism_ratio": 0.05
    }

    probabilities ={
        "prob_crossover": 0.01
        ,"prob_technique_mutation": 0.05
        ,"prob_magnitude_mutation": 0.05
        ,"prob_probability_mutation": 0.05
    }

    requirements = {
        "num_crossovers_required": 0
        ,"num_technique_mutation_required": 0
        ,"num_magnitude_mutation_required": 0
        ,"num_probability_mutation_required": 1
    }

    results_rows = [",".join(["num_sub_policies (N)","num_technique (Q)","prob_crossover","prob_technique_mutation","prob_magnitude_mutation","prob_probability_mutation","probability_of_outcome"])]

    for prob_cross in np.arange(0.001,0.11, 0.001):
        print("prob_cross",prob_cross)
        for prob_tech in np.arange(0.001,0.11, 0.001):
            print("prob_tech",prob_tech)
            for prob_mag in np.arange(0.001,0.11, 0.001):
                for prob_pro in np.arange(0.001,0.11, 0.001):
                
                    probabilities ={
                        "prob_crossover": prob_cross
                        ,"prob_technique_mutation": prob_tech
                        ,"prob_magnitude_mutation": prob_mag
                        ,"prob_probability_mutation": prob_pro
                    }

                    probability_of_outcome = CalculateProbabilityOfOutcome(requirements, probabilities, population_attributes)
                    

                    results_rows.append(",".join( [str(x) for x in [population_attributes["num_sub_policies"], population_attributes["num_technique"],prob_cross,prob_tech,prob_mag,prob_pro,probability_of_outcome]] ) )            
                    # print("prob_crossove:", prob_cross)
                    # print("Outcome Probability: "+str(probability_of_outcome))
    

    with open("prob_results.csv","w") as f:
        f.write("\n".join(results_rows))
                