import json
import torch
import numpy as np
from model import DeepT, DeepTConceptLearner
from ontolearn import KnowledgeBase
from ontolearn.metrics import F1
from ontolearn.util import get_full_iri
from ontolearn.refinement_operators import CustomRefinementOperator
from ontolearn import *
from ontolearn.metrics import F1
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.stats import describe

file_path = 'Log/2020-06-30 10:56:46.174980/'

with open(file_path + 'Testing.json', 'r') as file_descriptor:
    concepts_test = json.load(file_descriptor)

f_scores_dl_foil = []

dl_foil_runtime = []

indexes = []
for _, problem in concepts_test.items():
    target_concept = problem['Target']
    print('\nTARGET CONCEPT:', target_concept)
    indexes.append(target_concept)
    p = problem['Positives']
    n = problem['Negatives']

    kb = KnowledgeBase(path='data/family-benchmark_rich_background.owl')
    model = CustomConceptLearner(knowledge_base=kb,
                                 refinement_operator=CustomRefinementOperator(kb=kb),
                                 quality_func=F1(),
                                 heuristic_func=DLFOILHeuristic(),
                                 search_tree=SearchTree(),
                                 terminate_on_goal=True,
                                 iter_bound=1_000,
                                 verbose=False)
    start_t = time.time()
    best_pred_dl_foild = model.predict(pos=p, neg=n)
    dl_foil_runtime.append(time.time() - start_t)
    f_scores_dl_foil.append(best_pred_dl_foild.quality)
    print('\t',best_pred_dl_foild)

f_scores_dl_foil = np.array(f_scores_dl_foil)
dl_foil_runtime = np.array(dl_foil_runtime)

print('F1 =>', describe(f_scores_dl_foil))
print('Runtime =>', describe(dl_foil_runtime))
