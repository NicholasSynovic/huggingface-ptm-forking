import json

with open('result.json', 'r') as file:
    data = json.load(file)

total = 0

for task, models in data.items():
    for item, modelName in models.items():
        if modelName and modelName != "null":
            total += 1

print("Total number of items:", total)