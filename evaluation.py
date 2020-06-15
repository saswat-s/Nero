import json
import torch
from model import DeepT

# TODO RELOAD PYTORCH MODEL
# TODO Perform prediction on any given positive and negative examples.

file_path = 'Log/2020-06-10 13:04:58.792332/'
model_path = 'kb.name_model.pt'

with open(file_path + 'parameters.json', 'r') as file_descriptor:
    param = json.load(file_descriptor)

model = DeepT(param)

model.load_state_dict(torch.load(file_path + model_path, map_location=torch.device('cpu')))
for parameter in model.parameters():
    parameter.requires_grad = False
model.eval()

print(model)
