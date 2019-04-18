import json


generations_data_path = "generation_genome_data.json"


generations_genome_json = None

with open(generations_data_path, "r") as f:
    generations_genome_json = json.loads(f.read())




sorted_generations_data_path = "generation_sorted_policies.json"

sorted_generations_genome_json = None

with open(sorted_generations_data_path, "r") as f:
	sorted_generations_genome_json = json.loads(f.read())



print("format_data")
