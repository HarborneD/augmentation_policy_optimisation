import json


sorted_generations_data_path = "generation_sorted_policies.json"

sorted_generations_genome_json = None

with open(sorted_generations_data_path, "r") as f:
	sorted_generations_genome_json = json.loads(f.read())

entries = []

for generation_i in range(len(sorted_generations_genome_json)):
    print(generation_i)
    generation_genomes = sorted_generations_genome_json[generation_i]

    genome_types = list(generation_genomes.keys())
    genome_types.sort()
    for genome_type in genome_types[:1]:
        print(genome_type)  
        for genome in generation_genomes[genome_type]:
            print(genome)  
            previous_genomes = ["all_genomes"]

            num_chromosome_in_genome = len(generation_genomes[genome_type][genome])

            for previous_genome in previous_genomes:
                entries.append( [previous_genome,genome_type,num_chromosome_in_genome] )

    output_string = "source,target,value\n"

    for entry in entries:
        output_string += ",".join([str(e) for e in entry]) +"\n"

    with open("generation_"+str(generation_i)+"sankey.csv", "w") as f:
        f.write(output_string)