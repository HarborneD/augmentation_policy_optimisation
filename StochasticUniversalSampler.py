import random

class StochasticUniversalSampler():

    def __init__(self,population_list):
        ### population_list in the form [(item, fitness)]
        self.population_list = population_list
        self.wheel = self.CreateWheel(population_list)


    def TotalFitness(self):
        return sum([x[1] for x in self.population_list])
    

    def CreateWheel(self, population):
        total_fitness = self.TotalFitness()

        if(total_fitness == 0):
            print(self.population_list)
        
        wheel = []
        current_wheel_position = 0
        for p in population:
            current_wheel_position += p[1]/float(total_fitness)
            wheel.append((current_wheel_position, p))
        
        #print("wheel:", wheel)
        return wheel

    
    def GenerateSelections(self,num_selections):
        pointers = self._CreatePointers(num_selections)

        return self._FindSelections(pointers)

    def _CreatePointers(self,num_pointers):
        pointer_start = random.random()

        pointer_space = 1.0/num_pointers
        #print("pointer_space",pointer_space)
        pointers = []

        pointer_num = pointer_start
        for p_i in range(num_pointers):
            #print("pointer_num",pointer_num)
            if(pointer_num > 1):
                pointer_num -= 1 
                #print("pointer_num",pointer_num)
                pointer_move = 0
            
            pointers.append(pointer_num)

            pointer_num += pointer_space
            
           
        pointers.sort()
        #print("pointers:", pointers)
        return pointers
    

    def _FindSelections(self,pointers):
        selections = []
        selections_found = 0
        for section in self.wheel:
            for pointer in pointers[selections_found:]:
                if(pointer < section[0]):
                    selections.append(section[1])
                    selections_found +=1
                else:
                    break
        

        return selections


if(__name__ == "__main__"):
    population = [("a",1),("b",2),("c",3),("d",4)]

    sus = StochasticUniversalSampler(population)

    selections = sus.GenerateSelections(2)

    print(selections)
        
