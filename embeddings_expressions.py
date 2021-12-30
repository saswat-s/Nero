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
from ontolearn import KnowledgeBase
from core.static_funcs import *
from core.loaders import *

def run(args):
    """

    :param args:
    :return:
    """

    def construct_expression_look_up(path):
        kb = KnowledgeBase(path=path,
                           reasoner_factory=ClosedWorld_ReasonerFactory)

        rho = SimpleRefinement(knowledge_base=kb)

        look_up_class_exp = {i.name: i for i in rho.atomic_class_expressions()}

        for i in rho.all_quantifiers():
            look_up_class_exp.setdefault(i.type, dict()).setdefault(i.role.name, dict()).update({i.filler.name: i})
        return look_up_class_exp

    # (3) Load the Pytorch Module.
    pre_trained_nero, _ = load_nero(args)
    look_up_class_exp = construct_expression_look_up(args.path_knowledge_base)
    selective_2Dplot(pre_trained_nero, [
        look_up_class_exp['existantial_quantifier_expression']['hasChild']['⊤'],  # close to Parent
        look_up_class_exp['Brother'],
        look_up_class_exp['Brother'] * look_up_class_exp['existantial_quantifier_expression']['hasSibling']['Female'],
        # look_up_class_exp['Brother'] * look_up_class_exp['exists']['hasSibling']['Male'],
        look_up_class_exp['Brother'] * look_up_class_exp['existantial_quantifier_expression']['hasSibling']['Mother'],
        look_up_class_exp['Brother'] * look_up_class_exp['existantial_quantifier_expression']['hasSibling']['Father'],
        look_up_class_exp['Brother'] * look_up_class_exp['universal_quantifier_expression']['hasSibling']['Father'],
        look_up_class_exp['Sister'],
        look_up_class_exp['Sister'] * look_up_class_exp['existantial_quantifier_expression']['hasSibling']['Female'],
        look_up_class_exp['Sister'] * look_up_class_exp['universal_quantifier_expression']['hasSibling']['Mother'],
        look_up_class_exp['Sister'] * look_up_class_exp['universal_quantifier_expression']['hasSibling']['Son'],
        look_up_class_exp['Sister'] * look_up_class_exp['universal_quantifier_expression']['hasSibling']['Son'] *
        look_up_class_exp['Mother'],
    ])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_of_experiment_folder", type=str,
                        default='Experiments/NeroFamily')

    parser.add_argument("--path_knowledge_base", type=str,
                        default='KGs/Family/Family.owl')
    parser.add_argument("--use_multiprocessing_at_parsing", type=int, default=0)
    parser.add_argument("--use_search", default=None)

    run(parser.parse_args())
