import os
from typing import Dict
import torch
import json
from core import NERO, DeepSet, ST, TargetClassExpression, f_measure
from random import randint
from argparse import ArgumentParser
import random
from core.static_funcs import ClosedWorld_ReasonerFactory
from ontolearn import KnowledgeBase


def run(args):
    kb = KnowledgeBase(path=args.path_knowledge_base,
                       reasoner_factory=ClosedWorld_ReasonerFactory)
    all_named_individuals = {i.get_iri().as_str() for i in kb.individuals()}
    problems = dict()
    if args.lp_gen_technique == 'Random':
        for _ in range(args.num_learning_problems):
            problems['RandomUnknown_' + str(_)] = {
                'positive_examples': list(set(random.sample(all_named_individuals, args.num_individuals_in_input_set))),
                'negative_examples': list(set(random.sample(all_named_individuals, args.num_individuals_in_input_set)))}
    else:
        raise ValueError

    with open(f'LP_{args.lp_gen_technique}_input_size_{args.num_individuals_in_input_set}.json', "w") as outfile:
        json.dump({'problems': problems}, outfile, indent=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_knowledge_base", type=str,
                        default=os.getcwd() + '/KGs/Mutagenesis/Mutagenesis.owl')
    # Neural related
    parser.add_argument("--lp_gen_technique", type=str,
                        default='Random')

    parser.add_argument("--num_learning_problems", type=int, default=50)
    parser.add_argument("--num_individuals_in_input_set", type=int, default=1)

    run(parser.parse_args())
