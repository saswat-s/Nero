"""
Deploy our approach
"""
import os
from typing import Dict
import torch
import json
from core import NERO, DeepSet, ST, TargetClassExpression, f_measure
from random import randint
from argparse import ArgumentParser
import random
import numpy as np
import pandas as pd
from core.static_funcs import *
from core.data_struct import ExpressionQueue
from core.static_funcs import ClosedWorld_ReasonerFactory
from core.dl_learner_binder import DLLearnerBinder


from ontolearn import KnowledgeBase
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing
import itertools
from typing import Iterable, Dict, List, Any
import pandas

"""
def search(settings):
    kb = KnowledgeBase(path=settings['path_knowledge_base'],
                       reasoner_factory=ClosedWorld_ReasonerFactory)

    def l(x):
        return [_ for _ in kb.reasoner().sub_classes(x, direct=True)]

    for i in l(kb.thing):
        print('top:', i)
        for j in l(i):
            print(j)

    for i in kb.all_individuals_set():
        print(i)
        for j in kb.reasoner().types(i):
            print(j)

    for i in itertools.chain(kb.ontology().object_properties_in_signature(),
                             kb.ontology().data_properties_in_signature()):
        print(i)

    kb = KnowledgeBase(path='KGs/Family/family-benchmark_rich_background.owl')
    NS = "http://www.benchmark.org/family#"

    manager = kb.ontology().get_owl_ontology_manager()
    # name of new class
    cls_a: OWLClass = OWLClass(IRI.create(NS, "MyEnrichedClass"))
    # example concept
    concept_a = OWLObjectUnionOf((OWLClass(IRI(NS, 'Brother')), OWLClass(IRI(NS, 'Father'))))
    manager.add_axiom(kb.ontology(), OWLEquivalentClassesAxiom(cls_a, concept_a))

    # save as new rdfxml file
    manager.save_ontology(kb.ontology(), IRI.create("file:/", "demir_family_new_onto.owl"))

    kb = KnowledgeBase(path='demir_family_new_onto.owl',
                       reasoner_factory=ClosedWorld_ReasonerFactory)

    print('asd')
    for i in kb.all_individuals_set():
        print(i)
        for j in kb.reasoner().types(i):
            print(j, end=', ')
    exit(1)

    def l(x):
        return [_ for _ in kb.reasoner().sub_classes(x, direct=True)]

    for i in l(kb.thing):
        for x in kb.reasoner().equivalent_classes(i):
            print(x, 'equivalen', i)
"""


def load_target_class_expressions_and_instance_idx_mapping(path_of_experiment_folder):
    """

    :param args:
    :return:
    """
    # target_class_expressions Must be empty and must be filled in an exactorder
    target_class_expressions = []
    df = pd.read_csv(path_of_experiment_folder + '/target_class_expressions.csv', index_col=0)
    for index, v in df.iterrows():
        t = TargetClassExpression(label_id=v['label_id'],
                                  name=v['name'],
                                  idx_individuals=eval(v['idx_individuals']),
                                  expression_chain=eval(v['expression_chain']))
        assert len(t.idx_individuals) == len(eval(v['idx_individuals']))

        target_class_expressions.append(t)
    """
    with open(path_of_experiment_folder + '/target_class_expressions.json', 'r') as f:
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
    """

    instance_idx_mapping = dict()
    with open(path_of_experiment_folder + '/instance_idx_mapping.json', 'r') as f:
        instance_idx_mapping.update(json.load(f))
    return target_class_expressions, instance_idx_mapping


def load_pytorch_module(args: Dict, path_of_experiment_folder) -> torch.nn.Module:
    """ Load weights and initialize pytorch module"""
    # (1) Load weights from experiment repo
    weights = torch.load(path_of_experiment_folder + '/final_model.pt', torch.device('cpu'))
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


def load_ncel(path_of_experiment_folder: str) -> NERO:
    # (1) Load the configuration setting.
    settings = dict()
    with open(path_of_experiment_folder + '/settings.json', 'r') as f:
        settings.update(json.load(f))

    # (2) Load target class expressions & instance_idx_mapping
    target_class_expressions, instance_idx_mapping = load_target_class_expressions_and_instance_idx_mapping(
        path_of_experiment_folder)
    # (1) Load Pytorch Module
    model = load_pytorch_module(settings, path_of_experiment_folder)

    model = NERO(model=model,
                 quality_func=f_measure,
                 target_class_expressions=target_class_expressions,
                 instance_idx_mapping=instance_idx_mapping)
    model.eval()
    return model


def predict(model, positive_examples, negative_examples):
    with torch.no_grad():
        return model.predict(str_pos=positive_examples, str_neg=negative_examples)


def report_model_results(results, name):
    results = pd.DataFrame.from_dict(results)
    print(
        f'{name}: F-measure:{results["F-measure"].mean():.3f}+-{results["F-measure"].std():.3f}\t'
        f'Runtime:{results["Runtime"].mean():.3f}+-{results["Runtime"].std():.3f}\t'
        f'NumClassTested:{results["NumClassTested"].mean():.3f}+-{results["NumClassTested"].std():.3f}'
    )


def run(args):
    path_knowledge_base = args.path_knowledge_base
    path_dl_learner = args.path_dl_learner
    # path_dl_learner=None
    path_of_json_learning_problems = args.path_of_json_learning_problems
    ncel_model = load_ncel(args.path_of_experiment_folder)

    lp = dict()
    with open(path_of_json_learning_problems, 'r') as f:
        lp.update(json.load(f))

    nero_results = []
    celoe_results = []
    eltl_results = []
    # Initialize models
    if path_dl_learner:
        celoe = DLLearnerBinder(binary_path=path_dl_learner, kb_path=path_knowledge_base, model='celoe')
        eltl = DLLearnerBinder(binary_path=path_dl_learner, kb_path=path_knowledge_base, model='eltl')
    # {'Prediction': 'Sister ⊔ (Female ⊓ (¬Granddaughter))', 'Accuracy': 0.7927, 'F-measure': 0.8283, 'NumClassTested': 3906, 'Runtime': 6.361}
    for target_str_name, v in lp['problems'].items():
        # OUR MODEL
        nero_results.append(
            ncel_model.fit(str_pos=v['positive_examples'], str_neg=v['negative_examples'],
                           topK=args.topK,
                           use_search=args.use_search, kb_path=args.path_knowledge_base))
        if path_dl_learner:
            celoe_results.append(celoe.fit(pos=v['positive_examples'], neg=v['negative_examples'],
                                           max_runtime=args.max_runtime_dl_learner).best_hypothesis())
            eltl_results.append(eltl.fit(pos=v['positive_examples'], neg=v['negative_examples'],
                                         max_runtime=args.max_runtime_dl_learner).best_hypothesis())
    # Later save into folder
    report_model_results(nero_results, name='NERO')
    if path_dl_learner:
        report_model_results(celoe_results, name='CELOE')
        report_model_results(eltl_results, name='ELTL')


if __name__ == '__main__':
    parser = ArgumentParser()
    # General
    # Repo Family
    # (1) Folder containing pretrained models
    folder_name = "Experiments"
    # (3) Evaluate NERO on Family benchmark dataset by using learning problems provided in DL-Learner

    # Path of an experiment folder
    parser.add_argument("--path_of_experiment_folder",
                        #default='PretrainedNero/NeroFamily'
                        default='Experiments/2021-12-17 18:25:04.430643'
                        )
    parser.add_argument("--path_knowledge_base")
    parser.add_argument("--path_of_json_learning_problems", default='LPs/Family/lp_dl_learner.json')
    # Inference Related
    parser.add_argument("--topK", type=int, default=100,
                        help='Test the highest topK target expressions')
    parser.add_argument("--path_dl_learner", type=str,
                        default=None
                        # default=os.getcwd() + '/dllearner-1.4.0/'
                        )
    parser.add_argument("--max_runtime_dl_learner", type=int, default=0)
    parser.add_argument('--use_search', default='None', help='Continues,IntersectNegatives,None')

    run(parser.parse_args())
