"""
Deploy our approach
"""
from typing import Dict

import pandas as pd
import torch
import json
import gradio as gr
from core import NERO, DeepSet, ST, TargetClassExpression, f_measure
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
                                      idx_individuals=frozenset(v['idx_individuals']),
                                      expression_chain=v['expression_chain'])
            assert len(t.idx_individuals) == len(v['idx_individuals'])

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


def load_ncel(args: Dict) -> NERO:
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


def launch_service(ncel_model):
    def predict(positive_examples, negative_examples, size_of_examples, random_examples: bool):
        if random_examples:
            # Either sample from here self.instance_idx_mapping
            # or sample from targets
            pos_str = random.sample(list(ncel_model.instance_idx_mapping.keys()), int(size_of_examples))
            neg_str = random.sample(list(ncel_model.instance_idx_mapping.keys()), int(size_of_examples))
        else:
            pos_str = positive_examples.split(",")
            neg_str = negative_examples.split(",")

        with torch.no_grad():
            results, run_time = ncel_model.predict(pos=pos_str, neg=neg_str, topK=100)
        if len(pos_str) < 20:
            s = f'E^+:{",".join(pos_str)}\nE^-:{",".join(neg_str)}\n'
        else:
            s = f'|E^+|:{len(pos_str)}\n|E^-|:{len(neg_str)}\n'
        values = []
        for ith, (f1, target_concept, str_instances, num_exp) in enumerate(results[:10]):
            # s += f'{ith + 1}. {target_concept.name}\t F1-score:{f1:.2f}\n'
            values.append([ith + 1, target_concept.name, round(f1, 3)])

        return s, pd.DataFrame(values, columns=['Rank', 'Exp.', 'F1-measure'])

    gr.Interface(
        fn=predict,
        inputs=[gr.inputs.Textbox(lines=5, placeholder=None, label='Positive Examples'),
                gr.inputs.Textbox(lines=5, placeholder=None, label='Negative Examples'),
                gr.inputs.Slider(minimum=1, maximum=100),
                "checkbox"],
        outputs=[gr.outputs.Textbox(label='Learning Problem'), gr.outputs.Dataframe(label='Predictions')],
        title='Rapid Induction of Description Logic Expressions via Nero',
        description='Click Random Examples & Submit.').launch()


def run(settings):
    with open(settings['path_of_experiments'] + '/settings.json', 'r') as f:
        settings.update(json.load(f))

    # (2) Load the Pytorch Module
    ncel_model = load_ncel(settings)

    launch_service(ncel_model)


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    parser.add_argument("--path_of_experiments", type=str, default=None)
    # Inference Related
    parser.add_argument("--topK", type=int, default=1000,
                        help='Test the highest topK target expressions')

    settings = vars(parser.parse_args())
    run(settings)
