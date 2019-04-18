import math


def CalcCapacity(n,w):
	return math.factorial(n) / (math.factorial(w) * math.factorial( n-w ) )


total_permutations = 0
for i in range(25):
	num_wild = i+1

	capacity = CalcCapacity(25,num_wild)
	print(capacity)
	total_permutations += capacity

print(total_permutations)


# for i in range(33554431):
# 	if i % 1000 == 0:
# 		print i


from itertools import compress, product

# def combinations(items):
#     return ( set(compress(items,mask)) for mask in 

masks = product(*[[0,1]]*25) 

# for m in masks:
# 	print(m)

print(len(list(masks)))