import os
import json

outputs_path = "/media/harborned/ShutUpN/repos/final_year_project/genetic_augment/outputs"

output_files = os.listdir(outputs_path)

valid_text = "valid_accuracy"
valid_text_length = len(valid_text)

fitness_dict = {}

for output_file in output_files[:]:
	output_file_path = os.path.join(outputs_path, output_file)

	output_text = ""

	with open(output_file_path , "r") as f:
		output_text = f.read()


	output_lines = output_text.split("\n")

	if(output_lines[-1] == ""):
		output_lines = output_lines[:-1]

	if(output_lines[-2][:valid_text_length] == valid_text):
		policy_id = output_lines[1].split(" ")[1]
		fitness = output_lines[-1].split(" ")[1]

		fitness_dict[policy_id] = float(fitness)

fitness_output_json_path = "policy_fitness.json"

with open(fitness_output_json_path, "w") as f:
	f.write(json.dumps(fitness_dict))


print(fitness_dict)