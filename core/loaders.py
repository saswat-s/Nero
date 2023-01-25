import torch
from typing import Dict
import time
import json
from .model import NERO
from .neural_arch import DeepSet
from .static_funcs import f_measure, ClosedWorld_ReasonerFactory,timeit
from .refinement_operator import SimpleRefinement
from ontolearn import KnowledgeBase
import pandas as pd


def load_target_class_expressions_and_instance_idx_mapping(args):
    # (1) Extract Class Expressions
    # target_class_expressions = []
    df = pd.read_csv(args.path_of_experiment_folder + '/target_class_expressions.csv', index_col=0)
    """
    if args.use_multiprocessing_at_parsing == 1:
        df = ddf.read_csv(args.path_of_experiment_folder + '/target_class_expressions.csv', dtype={'label_id': 'int',
                                                                                                   'name': 'object',
                                                                                                   'str_individuals': 'object',
                                                                                                   'idx_individuals': 'object',
                                                                                                   'atomic_expression': 'object',
                                                                                                   'concepts': 'object',
                                                                                                   'filler': 'object',
                                                                                                   'role': 'object',
                                                                                                   })
        df = df.compute(scheduler='processes').set_index('Unnamed: 0')
    else:
        df = pd.read_csv(args.path_of_experiment_folder + '/target_class_expressions.csv', index_col=0)
    """
    if args.use_search is not None:
        target_class_expressions = df  # df.filter(['label_id', 'type','name', 'str_individuals', 'idx_individuals','expression_chain','length'])
        """
        print('Parse KG')
        kb = KnowledgeBase(path=args.path_knowledge_base,
                           reasoner_factory=ClosedWorld_ReasonerFactory)
        rho = SimpleRefinement(knowledge_base=kb)
        for index, v in df.iterrows():
            rho.dict_to_exp(v.to_dict())
            continue
        target_class_expressions.append(t)
        """
        """
            if v['type'] == 'atomic_expression':
                t = rho.expression[v['name']]
                t.label_id = int(v['label_id'])
                t.idx_individuals = eval(v['idx_individuals'])
            elif v['type'] == 'negated_expression':
                t = rho.expression[v['name']]
                t.label_id = int(v['label_id'])
                t.idx_individuals = eval(v['idx_individuals'])
            elif v['type'] == 'intersection_expression':
                t = IntersectionClassExpression(label_id=v['label_id'],
                                                name=v['name'], length=length_info,
                                                str_individuals=eval(v['str_individuals']),
                                                idx_individuals=eval(v['idx_individuals']),
                                                expression_chain=eval(v['expression_chain']))
            elif v['type'] == 'union_expression':
                t = UnionClassExpression(label_id=v['label_id'],
                                         name=v['name'], length=length_info,
                                         str_individuals=eval(v['str_individuals']),
                                         idx_individuals=eval(v['idx_individuals']),
                                         expression_chain=eval(v['expression_chain']))
            elif v['type'] == 'existantial_quantifier_expression':
                t = ExistentialQuantifierExpression(label_id=v['label_id'],
                                                    name=v['name'],
                                                    str_individuals=eval(v['str_individuals']),
                                                    idx_individuals=eval(v['idx_individuals']),
                                                    expression_chain=eval(v['expression_chain']))
            elif v['type'] == 'universal_quantifier_expression':
                t = UniversalQuantifierExpression(label_id=v['label_id'],
                                                  name=v['name'],
                                                  str_individuals=eval(v['str_individuals']),
                                                  idx_individuals=eval(v['idx_individuals']),
                                                  expression_chain=eval(v['expression_chain']))
            else:
                print(v['type'])
                raise ValueError
            assert len(t.idx_individuals) == len(eval(v['idx_individuals']))
            """
    else:
        target_class_expressions = df  # df.filter(['label_id', 'type','name', 'str_individuals', 'idx_individuals','expression_chain','length'])
        """
        for index, v in df.iterrows():
            target_class_expressions.append(TargetClassExpression(label_id=v['label_id'],
                                                                  name=v['name'],
                                                                  str_individuals=eval(v['str_individuals']),
                                                                  idx_individuals=eval(v['idx_individuals']),
                                                                  ))
        """
    instance_idx_mapping = dict()
    with open(args.path_of_experiment_folder + '/instance_idx_mapping.json', 'r') as f:
        instance_idx_mapping.update(json.load(f))

    return target_class_expressions, instance_idx_mapping


def load_pytorch_module(args: Dict, path_of_experiment_folder) -> torch.nn.Module:
    """ Load weights and initialize pytorch module"""
    # (1) Load weights from experiment repo
    weights = torch.load(path_of_experiment_folder + '/final_model.pt', torch.device('cpu'))
    if args['neural_architecture'] == 'DeepSet':
        model = DeepSet(args)
    else:
        raise NotImplementedError('There is no other model')
    model.load_state_dict(weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

@timeit
def load_nero(args):
    """
    :param args: Namespace
    :return:
    """
    start_time = time.time()
    # (1) Load the configuration setting.
    settings = dict()
    with open(args.path_of_experiment_folder + '/settings.json', 'r') as f:
        settings.update(json.load(f))
    # (2) Load target class expressions & instance_idx_mapping
    target_class_expressions, instance_idx_mapping = load_target_class_expressions_and_instance_idx_mapping(args)
    if args.use_search == 'SmartInit':
        kb_path = args.path_knowledge_base
    else:
        kb_path = None
    # (3) Load Pytorch Module
    model = NERO(model=load_pytorch_module(settings, args.path_of_experiment_folder),
                 quality_func=f_measure,
                 target_class_expressions=target_class_expressions,
                 instance_idx_mapping=instance_idx_mapping,
                 kb_path=kb_path)
    model.eval()
    return model, time.time() - start_time
