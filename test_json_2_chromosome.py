from evaluator import augmentation_transforms
from GeneticAugment import FetchSubPolicyFromIndex,CreateSpeciesAttributeDict,BuildBlankPolicyChromosome,MapLocalAugmentationIndexsToGlobalInChromosome, MapLocalMagnitudeIndexsToGlobalInChromosome, MapLocalProbabilityIndexsToGlobalInChromosome

import json

augmentation_list = list(augmentation_transforms.TRANSFORM_NAMES)

chromosome = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.9, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.7, 0.8, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.4, 0.6, 0.2]

policy_json = {
    "policy": {
        "policy": [
            [
                [
                    "FlipUD",
                    0.4,
                    6
                ],
                [
                    "Color",
                    0.0,
                    0
                ]
            ],
            [
                [
                    "FlipLR",
                    0.0,
                    0
                ],
                [
                    "ShearY",
                    0.8,
                    7
                ]
            ],
            [
                [
                    "Smooth",
                    0.3,
                    2
                ],
                [
                    "Sharpness",
                    0.9,
                    8
                ]
            ],
            [
                [
                    "Brightness",
                    0.7,
                    4
                ],
                [
                    "Cutout",
                    0.8,
                    5
                ]
            ],
            [
                [
                    "Posterize",
                    0.1,
                    6
                ],
                [
                    "AutoContrast",
                    0.4,
                    2
                ]
            ]
        ],
        "id": "test_policyJSON_exp_0001_20e_10p_5-2_00000_00000"
    }
}


num_technqiues_per_sub_policy = 2
num_sub_policies_per_policy = 5

population_size = 1

    
species_attributes = CreateSpeciesAttributeDict(population_size,len(augmentation_list), num_technqiues_per_sub_policy, num_sub_policies_per_policy)
    


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


print(chromosome)
print(loaded_chromosome)

for i in range(len(chromosome)):
    print(str(chromosome[i])+ " " + str(loaded_chromosome[i]))


# paper_cifar_10_policy = [
#   [('Invert',0.1,7),('Contrast',0.2,6)],
#   [('Rotate',0.7,2),('TranslateX',0.3,9)],
#   [('Sharpness',0.8,1),('Sharpness',0.9,3)],
#   [('ShearY',0.5,8),('TranslateY',0.7,9)],
#   [('AutoContrast',0.5,8),('Equalize',0.9,2)],
#   [('ShearY',0.2,7),('Posterize',0.3,7)],
#   [('Color',0.4,3),('Brightness',0.6,7)],
#   [('Sharpness',0.3,9),('Brightness',0.7,9)],
#   [('Equalize',0.6,5),('Equalize',0.5,1)],
#   [('Contrast',0.6,7),('Sharpness',0.6,5)],
#   [('Color',0.7,7),('TranslateX',0.5,8)],
#   [('Equalize',0.3,7),('AutoContrast',0.4,8)],
#   [('TranslateY',0.4,3),('Sharpness',0.2,6)],
#   [('Brightness',0.9,6),('Color',0.2,8)],
#   [('Solarize',0.5,2),('Invert',0.0,3)],
#   [('Equalize',0.2,0),('AutoContrast',0.6,0)],
#   [('Equalize',0.2,8),('Equalize',0.6,4)],
#   [('Color',0.9,9),('Equalize',0.6,6)],
#   [('AutoContrast',0.8,4),('Solarize',0.2,8)],
#   [('Brightness',0.1,3),('Color',0.7,0)],
#   [('Solarize',0.4,5),('AutoContrast',0.9,3)],
#   [('TranslateY',0.9,9),('TranslateY',0.7,9)],
#   [('AutoContrast',0.9,2),('Solarize',0.8,3)],
#   [('Equalize',0.8,8),('Invert',0.1,3)],
#   [('TranslateY',0.7,9),('AutoContrast',0.9,1)],
# ]

# paper_policy_json = {
#     "policy": {
#         "id": "AutoAugmentBestCifar10Policy",
#         "policy": []
#     }
# }

# for sub_policy in paper_cifar_10_policy:
#     paper_policy_json["policy"]["policy"].append([list(t) for t in sub_policy])


# paper_json_string = json.dumps(paper_policy_json, indent=4)

# with open("autoaugment_paper_cifar10.json","w") as f:
#     f.write(paper_json_string)
            