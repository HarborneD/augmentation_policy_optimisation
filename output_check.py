import os


output_dir = "outputs"

output_files = os.listdir(output_dir)

accuracies = []
print(len(output_files))
for output_file in output_files[:]:
	file_path = os.path.join(output_dir,output_file)

	final_accuracy = ""
	with open(file_path,"r") as f:
		file_text = f.read()
		text_lines = file_text.split("\n")
		if(len(text_lines) < 2):
			continue
		final_accuracy = text_lines[-2].split(" ")[-1]

	accuracies.append( (output_file, final_accuracy, text_lines) )

highest_accuracy = 0
highest_output = ""
for accuracy in accuracies:
	try:
		accuracy_val = float(accuracy[1])
	except:
		print("failed to find accuracy:")
		# print(accuracy)
	if(accuracy_val > float(highest_accuracy)):
		highest_accuracy = accuracy[1]
		highest_output = accuracy[0]

accuracies = sorted(accuracies,key=lambda x: x[1],reverse=True)

print(highest_output)
print(highest_accuracy)

output_lines = [ ",".join(["output_file", "policy_id","accuracy" ]) ]

for accuracy in accuracies:
	output_lines.append( ",".join([ accuracy[0],str(accuracy[2][1]), str(accuracy[1]) ]) )


with open("outputs_accumulated.csv" , "w") as f:
	f.write( "\n".join(output_lines) )