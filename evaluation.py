import json
import torch
import numpy as np
from model import DeepT, DeepTConceptLearner
from ontolearn import KnowledgeBase,SearchTree
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
with open(file_path + 'seen_concepts.json', 'r') as file_descriptor:
    seen_concepts = json.load(file_descriptor)

kb_deep = KnowledgeBase(path=file_path + 'enriched.owl')

for k, concept in kb_deep.str_to_concept_from_iterable(seen_concepts.keys()).items():
    if len(concept.instances) == 0:
        concept.instances={kb_deep.str_to_indv_obj[i] for i in seen_concepts[k]}



f_scores_deept_t = []
deept_t_runtime = []

indexes = []
for _, problem in concepts_test.items():
    target_concept = problem['Target']
    print('\nTARGET CONCEPT:', target_concept)
    indexes.append(target_concept)
    p = problem['Positives']
    n = problem['Negatives']

    model = DeepTConceptLearner(file_path=file_path,
                                knowledge_base=kb_deep,
                                refinement_operator=CustomRefinementOperator(kb=kb_deep),
                                search_tree=SearchTree(),
                                quality_func=F1(),
                                terminate_on_goal=True,
                                iter_bound=1000,
                                verbose=False)
    start_t = time.time()
    best_pred_deep_t = model.predict(pos=p, neg=n)
    deept_t_runtime.append(time.time() - start_t)

    f_scores_deept_t.append(best_pred_deep_t.quality)

f_scores_deept_t = np.array(f_scores_deept_t)
deept_t_runtime = np.array(deept_t_runtime)

print('F1 =>', describe(f_scores_deept_t))
print('Runtime =>', describe(deept_t_runtime))
