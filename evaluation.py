import json
import torch
import numpy as np
from model import DeepT
from helper_func import score_with_labels
# TODO RELOAD PYTORCH MODEL
# TODO Perform prediction on any given positive and negative examples.

file_path = 'Log/2020-06-26 17:24:19.798027/'
model_path = 'kb.name_model.pt'

with open(file_path + 'parameters.json', 'r') as file_descriptor:
    param = json.load(file_descriptor)

model = DeepT(param)

model.load_state_dict(torch.load(file_path + model_path, map_location=torch.device('cpu')))
for parameter in model.parameters():
    parameter.requires_grad = False
model.eval()

with open(file_path + 'Testing.json', 'r') as file_descriptor:
    concepts_test = json.load(file_descriptor)

with open(file_path + 'index.json', 'r') as file_descriptor:
    index = json.load(file_descriptor)

for _, problem in concepts_test.items():
    target_concept = problem['Target']
    print(target_concept)
    print(problem.keys())

    pos = problem['Positives']
    neg = problem['Negatives']


    assert len(pos)==len(neg)

    true_f1_scores = []
    predicted_f_dist = []
    f1_score_to_report = []

    with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):


        k_inputs = []
        k_f_measures_per_label = []
        k_pred = []
        for k in range(10):
            try:

                k_f_measures_per_label.append(score_with_labels(pos=pos, neg=neg, labels=labels))
                k_inputs.append([index[i] for i in pos + neg])
                # input = torch.tensor(input).reshape(1, len(input))
            except ValueError:
                continue
        k_inputs = torch.tensor(np.array(k_inputs), dtype=torch.int64)

        if len(k_inputs) == 0:
            raise ValueError

        predictions = model.forward(k_inputs)

        averaged_predicted_f1_dist = torch.mean(predictions, dim=0)

        # We use averaging as we can not make use of all individuals.
        # Save average predicted F-1 score distribution and average TRUE F1-scores
        predicted_f_dist.append(averaged_predicted_f1_dist.numpy())
        true_f1_scores.append(np.array(k_f_measures_per_label).mean(axis=0))

        # Save average predicted F-1 score distribution.
        values, indexes = torch.topk(averaged_predicted_f1_dist, 2)

        best_pred = labels[indexes]

        x_pos, x_neg = set(pos), set(neg)

        # TODO calculate F1-score

exit(1)

true_f1_scores = []
predicted_f_dist = []
f1_score_to_report = []
with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    for true_concept in concepts_test:
        k_inputs = []
        k_f_measures_per_label = []
        k_pred = []
        for k in range(params['num_of_samples_for_prediction_averaging']):
            try:
                x_pos, x_neg = data_generator.pos_neg_sampling_from_concept(true_concept,
                                                                            params['num_of_inputs_for_model'])
                k_f_measures_per_label.append(data_generator.score_with_labels(pos=x_pos, neg=x_neg, labels=labels))
                k_inputs.append([data_generator.indx[i] for i in x_pos + x_neg])
                # input = torch.tensor(input).reshape(1, len(input))
            except ValueError:
                continue
        k_inputs = torch.tensor(np.array(k_inputs), dtype=torch.int64)

        if len(k_inputs) == 0:
            print('Can not test {0} num_instance:{1}'.format(true_concept, len(true_concept.instances)))
            continue

        predictions = model.forward(k_inputs)

        averaged_predicted_f1_dist = torch.mean(predictions, dim=0)

        # We use averaging as we can not make use of all individuals.
        # Save average predicted F-1 score distribution and average TRUE F1-scores
        predicted_f_dist.append(averaged_predicted_f1_dist.numpy())
        true_f1_scores.append(np.array(k_f_measures_per_label).mean(axis=0))

        # Save average predicted F-1 score distribution.
        values, indexes = torch.topk(averaged_predicted_f1_dist, 2)

        best_pred = labels[indexes]

        logger.info('Top {0} Predictions for true concept:{1} num_instance:{2}'.format(len(best_pred), true_concept.str,
                                                                                       len(true_concept.instances)))

        # TODO APPLY CONCEPT LEARNING.
        for ith, pred in enumerate(best_pred):
            f_1 = data_generator.score_with_instances(pos=true_concept.instances,
                                                      neg=kb.thing.instances - true_concept.instances,
                                                      instances=pred.instances)

            if ith == 0:
                f1_score_to_report.append(f_1)

            logger.info(
                '{0}.th {1} with num_instance:{2}\tF-1 score:{3}'.format(ith + 1, pred.str, len(pred.instances), f_1))

f1_score_to_report = np.array(f1_score_to_report)
logger.info(
    'Mean and STD of F-1 score of 1.th predictions at testing:{0:.3f} +- {1:.3f}'.format(f1_score_to_report.mean(),
                                                                                         f1_score_to_report.std()))
