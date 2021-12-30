"""
Deploy our approach
"""
import os
import time
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
from core.loaders import *
#from core.data_struct import ExpressionQueue
#from core.static_funcs import ClosedWorld_ReasonerFactory,load_ncel
from core.dl_learner_binder import DLLearnerBinder

from ontolearn import KnowledgeBase
from owlapy.model import OWLEquivalentClassesAxiom, OWLClass, IRI, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectSomeValuesFrom, OWLObjectInverseOf, OWLObjectProperty, OWLThing
import itertools
from typing import Iterable, Dict, List, Any
import pandas
from core.expression import *

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
    path_dl_learner = None
    path_of_json_learning_problems = args.path_of_json_learning_problems
    print('Model loading')
    ncel_model, loading_time_to_add = load_nero(args)
    print(f'Model loaded:{loading_time_to_add}')

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
    print('Iterating over learning problems')
    for target_str_name, v in lp['problems'].items():
        report = ncel_model.fit(str_pos=v['positive_examples'], str_neg=v['negative_examples'],
                                topK=args.topK,
                                use_search=args.use_search, kb_path=args.path_knowledge_base)
        # OUR MODEL
        nero_results.append(report)
        if path_dl_learner:
            celoe_results.append(celoe.fit(pos=v['positive_examples'], neg=v['negative_examples'],
                                           max_runtime=args.max_runtime_dl_learner).best_hypothesis())
            eltl_results.append(eltl.fit(pos=v['positive_examples'], neg=v['negative_examples'],
                                         max_runtime=args.max_runtime_dl_learner).best_hypothesis())
    # Also consider preprocessing time for Nero
    report['Runtime'] += loading_time_to_add
    report_model_results(nero_results, name='NERO')
    if path_dl_learner:
        report_model_results(celoe_results, name='CELOE')
        report_model_results(eltl_results, name='ELTL')


if __name__ == '__main__':
    parser = ArgumentParser()
    # Path of an experiment folder
    parser.add_argument("--path_of_experiment_folder",
                        #default='Best/NeroFamily',
                        default='Best/NeroMutagenesis',
                        )
    parser.add_argument("--path_knowledge_base",
                        #default='KGs/Family/Family.owl',
                        default='KGs/Mutagenesis/Mutagenesis.owl'
                        )
    parser.add_argument("--path_of_json_learning_problems",
                        #default='LPs/Family/lp_dl_learner.json'
                        default='LPs/Mutagenesis/lp_dl_learner.json'
                        )
    # Inference Related
    parser.add_argument("--topK", type=int, default=100,
                        help='Test the highest topK target expressions')
    parser.add_argument("--use_multiprocessing_at_parsing", type=int,
                        default=1, help='1 or 0')
    parser.add_argument("--path_dl_learner", type=str,
                        default=None
                        # default=os.getcwd() + '/dllearner-1.4.0/'
                        )
    parser.add_argument("--max_runtime_dl_learner", type=int, default=10)
    parser.add_argument('--use_search', default=None, help='Continues,SmartInit')

    run(parser.parse_args())
