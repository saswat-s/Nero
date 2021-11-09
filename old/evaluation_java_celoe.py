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

file_path = 'Log/2020-07-02 11:03:41.004878/'
with open(file_path + 'Testing.json', 'r') as file_descriptor:
    concepts_test = json.load(file_descriptor)

knowledge_base_path = '/home/demir/Desktop/DeepTunnellingForRefinementOperators/data/family-benchmark_rich_background.owl'

dl_leaner = DLLearnerBinder()  # send the path
indexes = []
for _, problem in concepts_test.items():
    target_concept = problem['Target']

    indexes.append(target_concept)
    positives = set(problem['Positives'])
    negatives = set(problem['Negatives'])
    print('\nTARGET CONCEPT:', target_concept)

    kb = KnowledgeBase(path='data/family-benchmark_rich_background.owl')
    model = CELOE(knowledge_base=kb,
                  refinement_operator=ModifiedCELOERefinement(kb=kb),# or LengthBaseRefinementOpt(kb=kb)
                  quality_func=F1(),
                  min_horizontal_expansion=2,
                  heuristic_func=CELOEHeuristic(),
                  search_tree=CELOESearchTree(),
                  terminate_on_goal=True,
                  iter_bound=1_000_000,
                  max_num_of_concepts_tested=500,
                  ignored_concepts={},
                  verbose=False)

    best_pred_celoe = model.predict(pos=positives, neg=negatives)



    # Create Config file
    # Run Config file
    str_best_concept_celoe, f_1score_celoe = dl_leaner.pipeline(knowledge_base_path=knowledge_base_path,
                                                    algorithm='celoe', positives=positives, negatives=negatives,
                                                    path_name=target_concept, num_of_concepts_tested=500)


    if f_1score_celoe / 100 != best_pred_celoe.quality:
        print('Java:CELOE:Best Prediction:{0}:\tF1-score:{1}'.format(str_best_concept_celoe, f_1score_celoe))

        print('Python:CELOE:Best Prediction:{0}:\tF1-score:{1} \tNumber of concepts tested:{2} '.format(best_pred_celoe, best_pred_celoe.quality,model.number_of_tested_concepts))

        """
        if f_1score / 100 > best_pred_celoe.quality:
            for i in model.search_tree:
                print(i)
            exit(1)
        """
