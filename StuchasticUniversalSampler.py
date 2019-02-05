import random

class StochasticUniversalSampler():

    def __init__(self,population_list):
        ### population_list in the form [(item, fitness)]
        self.population_list = population_list
    


    def TotalFitness(self):
        return sum([x[1] for x in self.population_list])
    
    def CreateWheel(self.)
