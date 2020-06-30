import json
from helper_classes import DLLearnerBinder
import torch
import numpy as np
from model import DeepT, DeepTConceptLearner
from ontolearn.util import get_full_iri
from ontolearn.refinement_operators import CustomRefinementOperator
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.stats import describe

from ontolearn import *

file_path = 'Log/2020-06-30 10:56:46.174980/'
with open(file_path + 'Testing.json', 'r') as file_descriptor:
    concepts_test = json.load(file_descriptor)

knowledge_base_path = '/home/demir/Desktop/DeepTunnellingForRefinementOperators/data/family-benchmark_rich_background.owl'

dl_leaner = DLLearnerBinder()  # send the path
indexes = []
for _, problem in concepts_test.items():
    target_concept = problem['Target']
    print('\nTARGET CONCEPT:', target_concept)
    indexes.append(target_concept)
    positives = problem['Positives']
    negatives = problem['Negatives']

    # Create Config file
    # Run Config file
    str_best_concept, f_1score = dl_leaner.pipeline(knowledge_base_path=knowledge_base_path,
                                                    algorithm='celoe', positives=positives, negatives=negatives,
                                                    path_name=target_concept, num_of_concepts_tested=200)


    kb = KnowledgeBase(path='data/family-benchmark_rich_background.owl')
    model = CELOE(knowledge_base=kb,
                  refinement_operator=ModifiedCELOERefinement(kb=kb),
                  quality_func=F1(),
                  min_horizontal_expansion=0,
                  heuristic_func=CELOEHeuristic(),
                  search_tree=CELOESearchTree(),
                  terminate_on_goal=True,
                  iter_bound=5_000,
                  num_of_concepts_tested=300,
                  ignored_concepts={},
                  verbose=False)

    best_pred_celoe = model.predict(pos=positives, neg=negatives)

    print('Best Prediction:{0}:\tF1-score:{1}'.format(str_best_concept, f_1score))
    print(best_pred_celoe)
