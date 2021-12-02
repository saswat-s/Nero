from typing import Dict

import matplotlib.pyplot as plt
import torch
import json
from core import NERO, DeepSet, ST, TargetClassExpression, f_measure
from random import randint
from argparse import ArgumentParser
import random
import numpy as np
import pandas as pd
from core.static_funcs import *
from sklearn.decomposition import PCA
import seaborn as sns

sns.set_style("whitegrid")


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
                                      idx_individuals=frozenset(v['idx_individuals']),
                                      expression_chain=v['expression_chain'])
            assert len(t.idx_individuals) == len(v['idx_individuals'])

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


def load_nero(args: Dict) -> NERO:
    # (2) Load target class expressions & instance_idx_mapping
    target_class_expressions, instance_idx_mapping = load_target_class_expressions_and_instance_idx_mapping(args)
    # (1) Load Pytorch Module
    model = load_pytorch_module(args)

    model = NERO(model=model,
                 quality_func=f_measure,
                 target_class_expressions=target_class_expressions,
                 instance_idx_mapping=instance_idx_mapping)
    model.eval()
    return model


def predict(model, positive_examples, negative_examples):
    with torch.no_grad():
        return model.predict(str_pos=positive_examples, str_neg=negative_examples)


def plot_image(length_2emb, length_2_uris, path_save_fig):
    low_emb = PCA(n_components=2).fit_transform(length_2emb)
    sns.regplot(x=low_emb[:, 0], y=low_emb[:, 1])
    for (n, emb) in zip(length_2_uris, low_emb):
        plt.annotate(n, (emb[0], emb[1]))
    """
    for (n, emb) in zip(length_2_uris, low_emb):
        if n in ['Granddaughter','Daughter','Sister','Grandmother','Mother','Female']:
            """"""
            plt.scatter(emb[0], emb[1], c='b')
        elif n in ['Grandfather','Grandson','Son','Brother','Father','Male']:
            """"""
            plt.scatter(emb[0], emb[1], c='b')
        else:
            plt.scatter(emb[0], emb[1], c='b')
        plt.annotate(n, (emb[0], emb[1]))
    """

    plt.ylim(-1000, 1100)
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.savefig('regplot' + path_save_fig)
    plt.plot()
    plt.show()


@torch.no_grad()
def run(settings):
    # (1) Load the configuration setting.
    with open(settings['path_of_experiment_folder'] + '/settings.json', 'r') as f:
        settings.update(json.load(f))

    instance_idx_mapping = dict()
    # (1) Load the configuration setting.
    with open(settings['path_of_experiment_folder'] + '/instance_idx_mapping.json', 'r') as f:
        instance_idx_mapping.update(json.load(f))

    id_to_str_individuals = dict(zip(instance_idx_mapping.values(), instance_idx_mapping.keys()))

    # (2) Load the Pytorch Module.
    pre_trained_nero = load_nero(settings)

    length_2_uris = []
    length_2emb = []

    length_3_uris = []
    length_3emb = []
    for tcl in pre_trained_nero.target_class_expressions:
        tcl: TargetClassExpression
        if len(tcl.name.split()) == 1 and ('¬' not in tcl.name):
            str_individuals = [id_to_str_individuals[_] for _ in tcl.idx_individuals]
            pos_emb = pre_trained_nero.positive_expression_embeddings(
                str_individuals).cpu().detach().numpy().flatten()
            length_2emb.append(pos_emb)
            length_2_uris.append(tcl.name)
        elif len(tcl.name.split()) == 3 and len(length_3emb) < 8 and ('⊓' in tcl.name) and ('¬' not in tcl.name):
            str_individuals = [id_to_str_individuals[_] for _ in tcl.idx_individuals]
            pos_emb = pre_trained_nero.positive_expression_embeddings(
                str_individuals).cpu().detach().numpy().flatten()
            length_3emb.append(pos_emb)
            length_3_uris.append(tcl.name)
        elif len(tcl.name.split()) == 3:
            """ Do nothing """
            print(tcl.name)
        else:
            """ Do nothing """

    plot_image(length_2emb, length_2_uris, 'family_plot.png')
    # plot_image(length_3emb, length_3_uris, 'complex_family_plot.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    # Repo Family
    parser.add_argument("--path_of_experiment_folder", type=str,
                        default='/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/PretrainedModels/Family/2021-11-17 18:00:28.803967')

    run(vars(parser.parse_args()))
