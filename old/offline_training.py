import json
import random
import torch
from torch.utils.data import DataLoader
from ontolearn import KnowledgeBase, CustomRefinementOperator
from ontolearn.util import create_experiment_folder, create_logger, get_full_iri
from ontolearn.metrics import F1
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import DeepT
import pandas as pd
import numpy as np
import umap
from helper_classes import TorchData, DataGeneratingProcess
from helper_func import score_with_labels
import ontolearn
from learner import LengthBaseLearner, LengthBaseRefinementOpt, SearchTreePriorityQueue

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

storage_path, _ = create_experiment_folder(folder_name='Log')
logger = create_logger(name='DeepT', p=storage_path)

path = 'data/family-benchmark_rich_background.owl'
# path = 'data/financial.owl'
# path = 'data/biopax.owl'
# path = 'data/NTNcombined.owl'
# path='data/humandisease3.owl'


max_concept_size_ratio = 1  # |T|*0.9
kb = KnowledgeBase(path=path, min_size_of_concept=1, max_concept_size_ratio=max_concept_size_ratio, verbose=True)
logger.info('Deep tunnelling for Refinement Operator'.format())
logger.info('Knowledgebase:{0}'.format(kb.name))

params = {
    'num_of_concepts_refined': 2,  # number of concepts to be refined.
    'max_concept_size_ratio': max_concept_size_ratio,
    'max_length_of_concept': 7,  # max length of concept
    'min_size_of_concept': 1,  # number of instances belonging to class
    'num_dim': 100,
    'num_of_epochs': 50,
    'batch_size': 1024,
    'num_of_inputs_for_model': 50,  # must be dividable by two , # TODO: Consider the given all individuals
    'num_of_times_sample_per_concept': 1,  # This should always 1.
    'flag_for_plotting': True
}

####################################### data generation process ########################################################
data_generator = DataGeneratingProcess(knowledge_base=kb,
                                       logger=logger,
                                       quality_func=F1(),
                                       refinement_operator=LengthBaseRefinementOpt(kb=kb),
                                       storage_path=storage_path)

concepts = data_generator.generate_concepts(num_of_concepts_refined=params['num_of_concepts_refined'],
                                            max_length_of_concept=params['max_length_of_concept'],
                                            min_size_of_concept=params['min_size_of_concept'])

concepts_train_split, concepts_test_split = train_test_split(concepts,
                                                             test_size=0.1, random_state=RANDOM_SEED)

concepts_seen_during_training = dict()
for c in concepts_train_split:
    concepts_seen_during_training[c.str] = list({get_full_iri(i) for i in c.instances})
with open(storage_path + '/seen_concepts.json', 'w') as jsonfile:
    json.dump(concepts_seen_during_training, jsonfile, indent=2)
del concepts_seen_during_training
kb.save(storage_path + '/enriched.owl', rdf_format="rdfxml")

concepts_test_split.sort(key=lambda x: len(x), reverse=True)
concepts_test_split = concepts_test_split[:25]

Target, X_test_pos, X_test_neg = data_generator.save(concepts_test_split, path=storage_path + '/Testing.json',
                                                     sample_size_for_pos_neg=params['num_of_inputs_for_model'])
# Important decision: Apply Jaccard, PPMI, etc
labels = np.array(concepts_train_split)
concept_labels = dict()
for index, concept in enumerate(labels):
    concept_labels[concept.str] = index

with open(storage_path + '/labels.json', 'w') as jsonfile:
    json.dump(concept_labels, jsonfile, indent=4)
del concept_labels

# Generate Training Data
targets, X, y = data_generator.convert_data(concepts_train_split, labels, params)
X = torch.tensor(X)
y = torch.tensor(y)  # F-scores
# y = torch.softmax(torch.tensor(y), dim=1)  # F-scores are turned into f-score distributions.

params['num_of_outputs'] = len(labels)
loader = DataLoader(TorchData(X, y), batch_size=params['batch_size'], shuffle=True, num_workers=1)
logger.info('Number of unique concepts in training split:{0}\tNumber of data-points {1}'.format(len(labels), len(X)))

params['num_instances'] = len(data_generator.indx)
model = DeepT(params)

with open(storage_path + '/parameters.json', 'w') as file_descriptor:
    temp = {'num_dim': params['num_dim'],
            'num_instances': params['num_instances'],
            'num_of_inputs_for_model': params['num_of_inputs_for_model'],
            'num_of_outputs': params['num_of_outputs']}
    json.dump(temp, file_descriptor, sort_keys=True, indent=3)
    del temp

model.init()
opt = torch.optim.Adam(model.parameters())
logger.info(model)
logger.info('Training starts: Number of data points:{0}'.format(len(X)))
loss_per_epoch = []
model.train()
for it in range(1, params['num_of_epochs'] + 1):

    running_loss = 0.0
    for (x_batch, y_batch) in loader:
        predictions = model.forward(x_batch)
        loss = model.loss(predictions, y_batch)
        loss.backward()
        opt.step()
        assert loss.item() > 0

        running_loss += loss.item()

    ##if it % 100 == 0:
    print('Loss at {0}.th epoch:{1}:'.format(it, loss.item()))

    loss_per_epoch.append(running_loss / len(loader))
###########################################################################################
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
torch.save(model.state_dict(), storage_path + '/kb.name' + '_model.pt')
logger.info('Training ends')

if params['flag_for_plotting']:
    plt.plot(loss_per_epoch)
    plt.grid(True)
    plt.savefig(storage_path + '/loss_history.pdf')
    plt.show()

    reducer = umap.UMAP()
    embeddings = model.state_dict()['embedding.weight']
    low_embd = reducer.fit_transform(embeddings)

    dijoint_concepts = []

    for _, v in kb.concepts.items():
        for __, vv in kb.concepts.items():
            if v.str != 'Nothing' and vv.str != 'Nothing':
                if len(v.instances & vv.instances) == 0:
                    dijoint_concepts.append((v, vv))

    random.shuffle(dijoint_concepts)
    for (a, b) in dijoint_concepts[:30]:
        selected_indx = [data_generator.indx[get_full_iri(i)] for i in a.instances]
        plt.scatter(low_embd[selected_indx, 0], low_embd[selected_indx, 1], c='r', label=a.str)

        selected_indx = [data_generator.indx[get_full_iri(i)] for i in b.instances]
        plt.scatter(low_embd[selected_indx, 0], low_embd[selected_indx, 1], c='b', label=b.str)

        title = '{0} vs {1}'.format(a.str, b.str)
        # plt.title(title)
        plt.legend()
        plt.savefig(storage_path + '/' + title + '.pdf')
        plt.show()

    """
    for k, v in kb.concepts.items():

        if 'Female' in v.str:
            selected_indx = [data_generator.indx[get_full_iri(i)] for i in v.instances]
            ax.scatter(low_embd[selected_indx, 0], low_embd[selected_indx, 1], c='r')
        elif 'Male' in v.str:
            selected_indx = [data_generator.indx[get_full_iri(i)] for i in v.instances]
            ax.scatter(low_embd[selected_indx, 0], low_embd[selected_indx, 1], c='b')
        else:
            pass
            # selected_indx = [data_generator.indx[get_full_iri(i)] for i in v.instances]
            # ax.scatter(low_embd[selected_indx, 0], low_embd[selected_indx, 1],c='c')

    plt.title('Female vs Male')
    plt.savefig(storage_path + '/ScatterPlotOfUMAP_EMB')
    plt.show()
    """

assert len(X_test_pos) == len(X_test_neg) == len(Target)
logger.info('Testing starts on:{0}'.format(len(X_test_pos)))
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.

true_f1_scores = []
predicted_f_dist = []
f1_score_to_report = []
with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    for target, x_pos, x_neg in zip(Target, X_test_pos, X_test_neg):
        set_x_pos, set_x_neg = set(x_pos), set(x_neg)

        idx_pos = [data_generator.indx[get_full_iri(i)] if i != data_generator.dummy_pos else data_generator.indx[i] for
                   i in list(x_pos) + [data_generator.dummy_pos for _ in
                                       range(params['num_of_inputs_for_model'] // 2 - len(x_pos))]]

        idx_neg = [data_generator.indx[get_full_iri(i)] if i != data_generator.dummy_neg else data_generator.indx[i] for
                   i in list(x_neg) + [data_generator.dummy_neg for _ in
                                       range(params['num_of_inputs_for_model'] // 2 - len(x_neg))]]

        x = torch.tensor(np.array(idx_pos + idx_neg), dtype=torch.int64)

        x = x.reshape(1, len(x))

        predictions = model.forward(x)

        values, indexes = torch.topk(predictions, 10)

        best_pred = labels[indexes]
        logger.info('Target:{0}'.format(target))
        for ith, pred in enumerate(best_pred):
            f_1 = data_generator.score_with_instances(pos=set_x_pos,
                                                      neg=set_x_neg,
                                                      instances=pred.instances)

            if ith == 0:
                f1_score_to_report.append(f_1)

            logger.info(
                '{0}.th {1} with num_instance:{2}\tPredicted F-1 score {3}\tTrue F-1 score:{4}'.format(ith + 1,
                                                                                                       pred.str, len(
                        pred.instances), values[ith], f_1))

f1_score_to_report = np.array(f1_score_to_report)
logger.info(
    'Mean and STD of F-1 score of 1.th predictions at testing:{0:.3f} +- {1:.3f}'.format(f1_score_to_report.mean(),
                                                                                         f1_score_to_report.std()))

exit(1)
"""

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
                k_f_measures_per_label.append(score_with_labels(pos=x_pos, neg=x_neg, labels=labels))
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

if params['flag_for_plotting']:
    true_f1_scores = pd.DataFrame(np.array(true_f1_scores), columns=[x.str for x in labels])

    plt.matshow(true_f1_scores.corr())
    # plt.title('Correlation of true quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(true_f1_scores.columns))], true_f1_scores.columns, rotation=90)
    plt.yticks([i for i in range(len(true_f1_scores.columns))], true_f1_scores.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of true quality of concepts at testing')
    plt.show()

    predicted_f_dist = pd.DataFrame(np.array(predicted_f_dist), columns=[x.str for x in labels])
    plt.matshow(predicted_f_dist.corr())
    # plt.title('Correlation of predicted quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(predicted_f_dist.columns))], predicted_f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(predicted_f_dist.columns))], predicted_f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of predicted quality of concepts at testing')
    plt.show()
"""""""""
"""
logger.info('Number of concepts in testing split: {0}'.format(len(X_test)))
logger.info('Testing starts')
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    predictions = model.forward(X_test)

    loss = model.loss(predictions.log(), y_test)

    logger.info('Loss at testing: {0}'.format(loss.item()))

    f_dist = pd.DataFrame(y_test.numpy(), columns=[x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    # plt.title('Correlation of true quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of true quality of concepts at testing')
    # plt.show()

    f_dist = pd.DataFrame(predictions.numpy(), columns=[x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    # plt.title('Correlation of predicted quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of predicted quality of concepts at testing')
    # plt.show()
exit(1)

X, y, kw = data.generate_data(**params)

params.update(kw)
logger.info('Hyperparameters:{0}'.format(params))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
logger.info('Number of concepts in training split:{0}'.format(len(X_train)))

model = DeepT(params)
model.init()
opt = torch.optim.Adam(model.parameters())

assert len(X_train) == len(y_train)  # for sanity check.

logger.info(model)
####################################################################################################################
loss_per_epoch = []
model.train()
for it in range(1, params['num_of_epochs'] + 1):
    running_loss = 0.0

    if it % 100 == 0:
        print(it)
    for j in range(0, len(X_train), params['batch_size']):
        opt.zero_grad()

        x_batch = X_train[j:j + params['batch_size']]
        y_batch = y_train[j:j + params['batch_size']]

        predictions = model.forward(x_batch)

        loss = model.loss(predictions.log(), y_batch)
        loss.backward()
        opt.step()
        assert loss.item() > 0

        running_loss += loss.item()
    loss_per_epoch.append(running_loss / len(X_train))
###########################################################################################
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
plt.plot(loss_per_epoch)
plt.grid(True)
plt.savefig(storage_path + '/loss_history.pdf')
# plt.show()

targets = []
inputs_ = []

logger.info('Number of concepts in testing split: {0}'.format(len(X_test)))
logger.info('Testing starts')
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.

with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    predictions = model.forward(X_test)

    loss = model.loss(predictions.log(), y_test)

    logger.info('Loss at testing: {0}'.format(loss.item()))

    f_dist = pd.DataFrame(y_test.numpy(), columns=[x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    # plt.title('Correlation of true quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of true quality of concepts at testing')
    # plt.show()

    f_dist = pd.DataFrame(predictions.numpy(), columns=[x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    # plt.title('Correlation of predicted quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of predicted quality of concepts at testing')
    # plt.show()
"""

"""


rho = CustomRefinementOperator(kb)

ratio = len(kb.thing.instances) * 1
# Generate concepts and prune those ones that do not satisfy the provided constraint.
concepts = [concept for concept in data.generate_concepts(**params) if
            ratio > len(concept.instances) > params['num_of_inputs_for_model']]
concepts_train, concepts_test = train_test_split(concepts, test_size=0.3, random_state=RANDOM_SEED)
"""

"""
# Important decision: Apply Jaccard, PPMI, etc
# labels = np.array(random.sample(concepts_train, 10))
labels = np.array(concepts)

params['num_of_outputs'] = len(labels)

# Generate Training Data
X, y = data.convert_data(concepts_train, labels, params)
X = torch.tensor(X)
y = torch.tensor(y)  # F-scores
# y = torch.softmax(torch.tensor(y), dim=1)  # F-scores are turned into f-score distributions.


dataloader = DataLoader(TorchData(X, y), batch_size=params['batch_size'], shuffle=True, num_workers=1)

logger.info('Number of unique concepts in training split:{0}\tNumber of data-points {1}'.format(len(labels), len(X)))

model = DeepT(params)

with open(storage_path + '/parameters.json', 'w') as file_descriptor:
    temp = {'num_dim': params['num_dim'],
            'num_instances': params['num_instances'],
            'num_of_inputs_for_model': params['num_of_inputs_for_model'],
            'num_of_outputs': params['num_of_outputs']}
    json.dump(temp, file_descriptor, sort_keys=True, indent=3)
    del temp

model.init()
opt = torch.optim.Adam(model.parameters())

logger.info(model)
####################################################################################################################
loss_per_epoch = []
model.train()
for it in range(1, params['num_of_epochs'] + 1):
    running_loss = 0.0

    if it % 100 == 0:
        print(it)

    for (x_batch, y_batch) in dataloader:
        predictions = model.forward(x_batch)
        loss = model.loss(predictions, y_batch)  # for KL
        loss.backward()
        opt.step()
        assert loss.item() > 0

        running_loss += loss.item()

    loss_per_epoch.append(running_loss / len(dataloader))
###########################################################################################
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
torch.save(model.state_dict(), storage_path + '/kb.name' + '_model.pt')

if params['flag_for_plotting']:
    plt.plot(loss_per_epoch)
    plt.grid(True)
    plt.savefig(storage_path + '/loss_history.pdf')
    plt.show()

    reducer = umap.UMAP()
    embeddings = model.state_dict()['embedding.weight']
    low_embd = reducer.fit_transform(embeddings)
    fig, ax = plt.subplots()
    ax.scatter(low_embd[:, 0], low_embd[:, 1])
    plt.savefig(storage_path + '/ScatterPlotOfUMAP_EMB')
    # for i, txt in enumerate(data.individuals):
    #    ax.annotate(txt, (low_embd[i, 0], low_embd[i, 1]))
    plt.show()

logger.info('Testing starts on:{0}'.format(len(concepts_test)))
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.

# params['num_of_times_sample_per_concept'] = 1

true_f1_scores = []
predicted_f_dist = []
f1_score_to_report = []
with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    for true_concept in concepts_test:
        k_inputs = []
        k_f_measures_per_label = []
        k_pred = []
        for k in range(10):
            try:
                x_pos, x_neg = data.pos_neg_sampling_from_concept(true_concept, params['num_of_inputs_for_model'])
                k_f_measures_per_label.append(data.score_with_labels(pos=x_pos, neg=x_neg, labels=labels))
                k_inputs.append([data.indx[i] for i in x_pos + x_neg])
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
        values, indexes = torch.topk(averaged_predicted_f1_dist, 3)

        best_pred = labels[indexes]

        logger.info('Top {0} Predictions for true concept:{1} num_instance:{2}'.format(len(best_pred), true_concept.str,
                                                                                       len(true_concept.instances)))
        for ith, pred in enumerate(best_pred):
            f_1 = data.score_with_instances(pos=true_concept.instances, neg=kb.thing.instances - true_concept.instances,
                                            instances=pred.instances)

            if ith == 0:
                f1_score_to_report.append(f_1)

            logger.info(
                '{0}.th {1} with num_instance:{2}\tF-1 score:{3}'.format(ith + 1, pred.str, len(pred.instances), f_1))

f1_score_to_report = np.array(f1_score_to_report)
logger.info(
    'Mean and STD of F-1 score of 1.th predictions at testing:{0:.3f} +- {1:.3f}'.format(f1_score_to_report.mean(),
                                                                                         f1_score_to_report.std()))

if params['flag_for_plotting']:
    true_f1_scores = pd.DataFrame(np.array(true_f1_scores), columns=[x.str for x in labels])

    plt.matshow(true_f1_scores.corr())
    # plt.title('Correlation of true quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(true_f1_scores.columns))], true_f1_scores.columns, rotation=90)
    plt.yticks([i for i in range(len(true_f1_scores.columns))], true_f1_scores.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of true quality of concepts at testing')
    plt.show()

    predicted_f_dist = pd.DataFrame(np.array(predicted_f_dist), columns=[x.str for x in labels])
    plt.matshow(predicted_f_dist.corr())
    # plt.title('Correlation of predicted quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(predicted_f_dist.columns))], predicted_f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(predicted_f_dist.columns))], predicted_f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of predicted quality of concepts at testing')
    plt.show()
##############################################################################################################
"""
"""
logger.info('Number of concepts in testing split: {0}'.format(len(X_test)))
logger.info('Testing starts')
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    predictions = model.forward(X_test)

    loss = model.loss(predictions.log(), y_test)

    logger.info('Loss at testing: {0}'.format(loss.item()))

    f_dist = pd.DataFrame(y_test.numpy(), columns=[x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    # plt.title('Correlation of true quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of true quality of concepts at testing')
    # plt.show()

    f_dist = pd.DataFrame(predictions.numpy(), columns=[x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    # plt.title('Correlation of predicted quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of predicted quality of concepts at testing')
    # plt.show()
exit(1)

X, y, kw = data.generate_data(**params)

params.update(kw)
logger.info('Hyperparameters:{0}'.format(params))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
logger.info('Number of concepts in training split:{0}'.format(len(X_train)))

model = DeepT(params)
model.init()
opt = torch.optim.Adam(model.parameters())

assert len(X_train) == len(y_train)  # for sanity check.

logger.info(model)
####################################################################################################################
loss_per_epoch = []
model.train()
for it in range(1, params['num_of_epochs'] + 1):
    running_loss = 0.0

    if it % 100 == 0:
        print(it)
    for j in range(0, len(X_train), params['batch_size']):
        opt.zero_grad()

        x_batch = X_train[j:j + params['batch_size']]
        y_batch = y_train[j:j + params['batch_size']]

        predictions = model.forward(x_batch)

        loss = model.loss(predictions.log(), y_batch)
        loss.backward()
        opt.step()
        assert loss.item() > 0

        running_loss += loss.item()
    loss_per_epoch.append(running_loss / len(X_train))
###########################################################################################
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.
plt.plot(loss_per_epoch)
plt.grid(True)
plt.savefig(storage_path + '/loss_history.pdf')
# plt.show()

targets = []
inputs_ = []

logger.info('Number of concepts in testing split: {0}'.format(len(X_test)))
logger.info('Testing starts')
model.eval()  # Turns evaluation mode on, i.e., dropouts are turned off.

with torch.no_grad():  # Important:    for j in range(0, len(X_train), params['batch_size']):

    predictions = model.forward(X_test)

    loss = model.loss(predictions.log(), y_test)

    logger.info('Loss at testing: {0}'.format(loss.item()))

    f_dist = pd.DataFrame(y_test.numpy(), columns=[x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    # plt.title('Correlation of true quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of true quality of concepts at testing')
    # plt.show()

    f_dist = pd.DataFrame(predictions.numpy(), columns=[x.str for x in data.labels])
    plt.matshow(f_dist.corr())
    # plt.title('Correlation of predicted quality of concepts at testing')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.xticks([i for i in range(len(f_dist.columns))], f_dist.columns, rotation=90)
    plt.yticks([i for i in range(len(f_dist.columns))], f_dist.columns)
    plt.gcf().set_size_inches(25, 25)
    plt.savefig(storage_path + '/Correlation of predicted quality of concepts at testing')
    # plt.show()
"""
