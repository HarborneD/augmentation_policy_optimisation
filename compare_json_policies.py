import json


def LoadPolicys(policy_path_list):
    policy_jsons = []

    for policy_path in policy_path_list:
        with open(policy_path, "r") as f:
            policy_jsons.append( json.loads(f.read())  )
    
    return policy_jsons


def SortPolicys(policy_list, sort_technique_index = 0):
    sorted_policys = []

    for policy in policy_list:
        sorted_policys.append(SortPolicy(policy,sort_technique_index))

    return sorted_policys


def SortPolicy(policy_json, sort_technique_index = 0):

    # for sub_policy_i in range(len(policy_json["policy"]["policy"])):
    #     policy_json["policy"]["policy"][sub_policy_i] = sorted(policy_json["policy"]["policy"][sub_policy_i], key=lambda x: (x[0],x[1],x[2]))
    

    policy_json["policy"]["policy"] = sorted(policy_json["policy"]["policy"], key=lambda x:(x[sort_technique_index][0],x[sort_technique_index][1],x[sort_technique_index][2]))

    return policy_json


def SortTechniquesInTechniqueDict(technique_sorted_dict):
    for policy_i in range(len(technique_sorted_dict)):
        for technique in technique_sorted_dict[policy_i]:
            technique_sorted_dict[policy_i][technique] = sorted(technique_sorted_dict[policy_i][technique], key=lambda x: (x[0],x[1],x[2]))

    return technique_sorted_dict


def SingleSortedSubPolicyDictToSingleSubPolicyList(sorted_sub_policy_dict):
    single_sorted_sub_policy_list = []

    for technique_key in sorted(list(sorted_sub_policy_dict.keys())):
        single_sorted_sub_policy_list += sorted_sub_policy_dict[technique_key]
    
    return single_sorted_sub_policy_list



def ProduceStaggeredRows(sorted_sub_policy_list, sort_technique_index = 0):
    num_sub_policies = 25

    staggered_rows = []

    included_sub_policies = 0
    max_sub_policies = 0 #len(list(sorted_sub_policy_list)) * num_sub_policies
    
    for policy_i in range(len(sorted_sub_policy_list)):
        max_sub_policies += len(sorted_sub_policy_list[policy_i])

    policy_index_track = [0] * len(list(sorted_sub_policy_list))

    while included_sub_policies < max_sub_policies:
        current_row_lowest_value = ["Z",11,11]
        current_row = []
        
        for policy_i in range(len(sorted_sub_policy_list)):
            sub_policy_i = policy_index_track[policy_i]
            if(sub_policy_i >= len(sorted_sub_policy_list[policy_i])):
                current_row.append("")
                continue
            
            if(sorted_sub_policy_list[policy_i][sub_policy_i][sort_technique_index] < current_row_lowest_value):
                current_row = ["" for i in range(policy_i)]  + [sorted_sub_policy_list[policy_i][sub_policy_i]]
                current_row_lowest_value = sorted_sub_policy_list[policy_i][sub_policy_i][sort_technique_index]
            elif (sorted_sub_policy_list[policy_i][sub_policy_i][sort_technique_index] == current_row_lowest_value):
                current_row.append(sorted_sub_policy_list[policy_i][sub_policy_i])
            else:
                current_row.append("")
        
        for policy_i in range(len(sorted_sub_policy_list)):
            if(current_row[policy_i] != ""):
                policy_index_track[policy_i] += 1
                included_sub_policies += 1


        staggered_rows.append(current_row)

    
    return staggered_rows


def ProduceStaggeredTechniqueRows(sorted_technique_list):
    num_sub_policies = 25

    staggered_rows = []

    included_sub_policies = 0
    max_sub_policies = len(list(sorted_technique_list)) * num_sub_policies
    
    policy_index_track = [0] * len(list(sorted_technique_list))

    while included_sub_policies < max_sub_policies:
        current_row_lowest_value = ["Z",11,11]
        current_row = []
        
        for policy_i in range(len(sorted_technique_list)):
            technique_i = policy_index_track[policy_i]
            if(technique_i >= len(sorted_technique_list[policy_i])):
                current_row.append("")
                continue

            if(sorted_technique_list[policy_i][technique_i] < current_row_lowest_value):
                current_row = ["" for i in range(policy_i)]  + [ [sorted_technique_list[policy_i][technique_i]] ]
                current_row_lowest_value = sorted_technique_list[policy_i][technique_i]
            elif (sorted_technique_list[policy_i][technique_i] == current_row_lowest_value):
                current_row.append( [ sorted_technique_list[policy_i][technique_i] ] )
            else:
                current_row.append("")
        
        for policy_i in range(len(sorted_technique_list)):
            if(current_row[policy_i] != ""):
                policy_index_track[policy_i] += 1
                included_sub_policies += 1


        staggered_rows.append(current_row)

    
    return staggered_rows


def OutputStaggeredRows(staggered_rows,output_path):
    output_rows = []

    for row in staggered_rows:
        output_rows.append(" || ".join( [format( " - ".join([str(y) for y in x]) , '<50')for x in row] ) )
        print( output_rows[-1])


    with open(output_path,"w") as f:
        f.write("\n".join(output_rows))


def ProduceSortedDicts(policy_list,longest_policy, sort_technique_index = 0):
    sub_policy_sorted_dict = [{} for i in range(len(policy_list))]
    technique_sorted_dict =  [{} for i in range(len(policy_list))]
    
    for i in range(longest_policy):
        print_row = []
        for policy_i in range(len(policy_list)):
            policy = policy_list[policy_i]

            if(i >= len(policy["policy"]["policy"])):
                continue

            sub_policy_key = policy["policy"]["policy"][i][sort_technique_index][0]
            
            if(sub_policy_key not in sub_policy_sorted_dict[policy_i]):
                sub_policy_sorted_dict[policy_i][sub_policy_key] = []

            sub_policy_sorted_dict[policy_i][sub_policy_key].append(policy["policy"]["policy"][i])
            print_row.append(policy["policy"]["policy"][i])
            for technique in policy["policy"]["policy"][i]:
                technique_key = technique[0]
                if(technique_key not in technique_sorted_dict[policy_i]):
                    technique_sorted_dict[policy_i][technique_key] = []
                
                technique_sorted_dict[policy_i][technique_key].append(technique)
        
        # print([print_row[i][0] for i in range(len(print_row))] )
        # print([print_row[i][1] for i in range(len(print_row))] )
        # print("")
    
    return sub_policy_sorted_dict, technique_sorted_dict



if __name__ == "__main__":
    policy_paths = [
        "autoaugment_paper_cifar10.json"
        # ,"policies_of_interest/pso_exp_0002_20e_10p_1-20_00196.json" 
        # ,"policies_of_interest/AutoAugmentBasedPopulation_exp_0002_120e_10p_25-2_00005_00005.json"
        ,"policies_of_interest/tabu_AutoAugmentBasedPopulation_exp_0001_20e_100ls_25-2_00001_00850.json"
    ]

    longest_policy = 25

    loaded_policy_list = LoadPolicys(policy_paths)

    policy_list = SortPolicys(loaded_policy_list)

    sub_policy_sorted_dict, technique_sorted_dicts = ProduceSortedDicts(policy_list,longest_policy)

    technique_sorted_dicts = SortTechniquesInTechniqueDict(technique_sorted_dicts)

    sorted_sub_policy_list = []

    for policy_i in range(len(sub_policy_sorted_dict)):
        sorted_sub_policy_list.append(SingleSortedSubPolicyDictToSingleSubPolicyList(sub_policy_sorted_dict[policy_i]))

    first_technique_staggered_rows = ProduceStaggeredRows(sorted_sub_policy_list)
    output_path = "first_technique_comparison.csv"
    OutputStaggeredRows(first_technique_staggered_rows,output_path)

    print("")
    print("____")
    print("")

    sorted_technique_list = []

    for policy_i in range(len(technique_sorted_dicts)):
        sorted_technique_list.append(SingleSortedSubPolicyDictToSingleSubPolicyList(technique_sorted_dicts[policy_i]))

    techniques_staggered_rows = ProduceStaggeredTechniqueRows(sorted_technique_list)
    output_path = "techniques_comparison.csv"
    OutputStaggeredRows(techniques_staggered_rows,output_path)

    print("")
    print("____")
    print("")

    policy_list = SortPolicys(loaded_policy_list,sort_technique_index=1)

    sub_policy_sorted_dict, technique_sorted_dicts = ProduceSortedDicts(policy_list,longest_policy, sort_technique_index = 1)
    
    sorted_sub_policy_list = []

    for policy_i in range(len(sub_policy_sorted_dict)):
        sorted_sub_policy_list.append(SingleSortedSubPolicyDictToSingleSubPolicyList(sub_policy_sorted_dict[policy_i]))

    second_technique_staggered_rows = ProduceStaggeredRows(sorted_sub_policy_list,sort_technique_index=1)
    output_path = "second_technique_comparison.csv"
    OutputStaggeredRows(second_technique_staggered_rows,output_path)

    print("")