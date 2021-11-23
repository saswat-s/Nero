"""
Deploy our approach
"""
from typing import Dict
import torch
import json
from core import NCEL, DeepSet, ST, TargetClassExpression, f_measure
from random import randint
from argparse import ArgumentParser
import random
import numpy as np


def load_target_class_expressions_and_instance_idx_mapping(args):
    """

    :param args:
    :return:
    """
    # target_class_expressions Must be empty and must be filled in an exactorder
    target_class_expressions = []
    with open(args['path_of_experiment_folder'] + '/target_class_expressions.json', 'r') as f:
        for k, v in json.load(f).items():
            k: str  # k denotes k.th label of target expression, json loading type conversion from int to str appreantly
            v: dict  # v contains info for Target Class Expression Object
            assert isinstance(k, str)
            assert isinstance(v, dict)
            try:
                k = int(k)
            except ValueError:
                print(k)
                print('Tried to convert to int')
                exit(1)
            try:

                assert k == v['label_id']
            except AssertionError:
                print(k)
                print(v['label_id'])
                exit(1)

            t = TargetClassExpression(label_id=v['label_id'],
                                      name=v['name'],
                                      individuals=frozenset(v['individuals']),
                                      idx_individuals=frozenset(v['idx_individuals']),
                                      expression_chain=v['expression_chain'])
            assert len(t.idx_individuals) == len(v['idx_individuals'])
            assert len(t.individuals) == len(v['individuals'])

            target_class_expressions.append(t)

    instance_idx_mapping = dict()
    with open(args['path_of_experiment_folder'] + '/instance_idx_mapping.json', 'r') as f:
        instance_idx_mapping.update(json.load(f))

    return target_class_expressions, instance_idx_mapping


def load_pytorch_module(args: Dict) -> torch.nn.Module:
    """ Load weights and initialize pytorch module"""
    # (1) Load weights from experiment repo
    weights = torch.load(args['path_of_experiment_folder'] + '/final_model.pt', torch.device('cpu'))
    if args['neural_architecture'] == 'DeepSet':
        model = DeepSet(args)
    elif args['neural_architecture'] == 'ST':
        model = ST(args)
    else:
        raise NotImplementedError('There is no other model')
    model.load_state_dict(weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def load_ncel(args: Dict) -> NCEL:
    # (2) Load target class expressions & instance_idx_mapping
    target_class_expressions, instance_idx_mapping = load_target_class_expressions_and_instance_idx_mapping(args)
    # (1) Load Pytorch Module
    model = load_pytorch_module(args)

    model = NCEL(model=model,
                 quality_func=f_measure,
                 target_class_expressions=target_class_expressions,
                 instance_idx_mapping=instance_idx_mapping)
    model.eval()
    return model


def predict(model, positive_examples, negative_examples):
    with torch.no_grad():
        return model.predict(pos=positive_examples, neg=negative_examples)


def run(settings):
    # (1) Load the configuration setting.
    with open(settings['path_of_experiment_folder'] + '/settings.json', 'r') as f:
        settings.update(json.load(f))

    # (2) Load the Pytorch Module.
    ncel_model = load_ncel(settings)

    # (3) Load Learning Problems
    lp = dict()
    with open(settings['path_of_json_learning_problems'], 'r') as f:
        lp.update(json.load(f))
    quality = []
    runtimes = []

    for target_str_name, v in lp['problems'].items():
        results, rt = ncel_model.predict(pos=v['positive_examples'], neg=v['negative_examples'], topK=settings['topK'])

        f1, target_concept = results[0]
        runtimes.append(rt)
        quality.append(f1)

    quality = np.array(quality)
    runtimes = np.array(runtimes)
    print(f'F-measure:{quality.mean():.3f} +- {quality.std():.3f} \t Runtimes: {runtimes.mean():.3f} +- {runtimes.std():.3f} in {len(lp["problems"])} learning problems')

if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_of_experiment_folder", type=str,
                        default='/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/PretrainedModels/Family/2021-11-17 18:00:28.803967')
    parser.add_argument("--path_of_json_learning_problems", type=str,
                        default='/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/LPs/Family/lp_dl_learner.json')
    # Inference Related
    parser.add_argument("--topK", type=int, default=None,
                        help='Test the highest topK target expressions')

    run(vars(parser.parse_args()))
