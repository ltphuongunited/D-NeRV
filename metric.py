import json
import os


logs = 'logs/UVG'
models = ['HDNeRV2', 'HDNeRV3']

for model in models:
    model_path = os.path.join(logs, model)
    for model in os.listdir(model_path):
        results_path = os.path.join(model_path, model, 'results.json')
        # print(results_path)
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                data = json.load(f)
            values_without_slash = {key: value for key, value in data.items() if '/' not in key}
            final_path = os.path.join(model_path, model, 'final.json')
            with open(final_path, 'w+') as json_file:
                json.dump(values_without_slash, json_file, indent=4)
