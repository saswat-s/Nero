"""
Deploy our approach
"""
from typing import Dict
import torch
import json
import gradio as gr
from core import NCEL, DeepSet, ST, TargetClassExpression, f_measure
from random import randint
from argparse import ArgumentParser
import random

def load_target_class_expressions_and_instance_idx_mapping(args):
    """

    :param args:
    :return:
    """
    # target_class_expressions Must be empty and must be filled in an exactorder
    target_class_expressions = []
    with open(args['path_of_experiments'] + '/target_class_expressions.json', 'r') as f:
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
    with open(args['path_of_experiments'] + '/instance_idx_mapping.json', 'r') as f:
        instance_idx_mapping.update(json.load(f))

    return target_class_expressions, instance_idx_mapping


def load_pytorch_module(args: Dict) -> torch.nn.Module:
    """ Load weights and initialize pytorch module"""
    # (1) Load weights from experiment repo
    weights = torch.load(args['path_of_experiments'] + '/final_model.pt', torch.device('cpu'))
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


def run(settings):
    with open(settings['path_of_experiments'] + '/settings.json', 'r') as f:
        settings.update(json.load(f))

    # (2) Load the Pytorch Module
    ncel_model = load_ncel(settings)

    def predict(positive_examples, negative_examples, topK, random_examples: bool):
        topK = int(round(topK))
        if random_examples:
            # Either sample from here self.instance_idx_mapping
            # or sample from targets
            pos = ncel_model.target_class_expressions[randint(0, len(ncel_model.target_class_expressions))].individuals
            neg = random.sample(list(ncel_model.instance_idx_mapping.keys()), len(pos))
        else:
            pos = positive_examples.split(",")
            neg = negative_examples.split(",")

        with torch.no_grad():
            results = ncel_model.predict(pos=pos, neg=neg, topK=topK)

        s = f' |E^+|{len(pos)},|E^+|{len(neg)}\n'
        for ith, (f1, target_concept) in enumerate(results[:10]):
            s += f'{ith + 1}. {target_concept.name}\t F1-score:{f1:.2f}\n'
        return s

    gr.Interface(
        fn=predict,
        inputs=[gr.inputs.Textbox(lines=5, placeholder=None, label=None),
                gr.inputs.Textbox(lines=5, placeholder=None, label=None),
                gr.inputs.Slider(minimum=10, maximum=1000), "checkbox"],
        outputs=[gr.outputs.Textbox(label='Class Expression Learning')]
    ).launch()


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_of_experiments", type=str,
                        default='/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/Experiments/2021-11-17 17:32:54.237217')
    # Inference Related
    parser.add_argument("--topK", type=int, default=1000,
                        help='Test the highest topK target expressions')

    settings = vars(parser.parse_args())
    run(settings)
