import os
import json
import random

#folder to store output file
outputFolder = 'samples'
os.makedirs(outputFolder, exist_ok=True)

#read JSON file
with open('result.json', 'r') as file:
    data = json.load(file)

#empty dictionary to store samples
sampledItems = {}

#percent of samples to get from each model task
samplePercentage = 0.1

#traverse JSON data and create list of samples
for task, items in data.items():
    
    models = {key: value for key, value in items.items()}
    
    #calculate the sample size based on desired percent of model items
    sampleSize = int(len(models) * samplePercentage)
    
    #sample items based on the calculated sample size
    sampledItems[task] = random.sample(list(models.items()), sampleSize)

# Save the sampled items into a .txt file
outputFilePath = os.path.join(outputFolder, 'sampledItems.txt')
with open(outputFilePath, 'w') as sample_file:
    for task, items in sampledItems.items():
        sample_file.write(task + '\n')
        for item in items:
            sample_file.write(f'  "{item[0]}": "{item[1]}"\n')

print(f'Sampled items saved in {outputFilePath}')

