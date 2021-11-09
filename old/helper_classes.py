import torch
from typing import Set

from owlready2 import destroy_entity
from torch.utils.data import Dataset
from ontolearn import Concept
from ontolearn.util import create_experiment_folder, create_logger, get_full_iri
from ontolearn.abstracts import AbstractScorer
from ontolearn.search import Node
import random
from typing import Iterable
import json
from helper_func import score_with_labels

import subprocess


class TorchData(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X[idx]
        y = self.y[idx]
        return x, y


class DataGeneratingProcess:
    """
    A class for constructing supervised learning problem.
    """

    def __init__(self, knowledge_base, logger=None,
                 refinement_operator=None,
                 quality_func=None,
                 storage_path=None
                 ):

        assert refinement_operator
        assert quality_func

        self.quality_func = quality_func
        self.kb = knowledge_base
        self.rho = refinement_operator
        if logger:
            self.logger = logger
            self.storage_path = storage_path
        else:
            self.storage_path, _ = create_experiment_folder(folder_name='../Log')
            self.logger = create_logger(name='Data', p=self.storage_path)

        self.individuals = list(self.kb.thing.instances)
        self.num_individuals = len(self.individuals)

        self.indx = dict()

        """
        self.indx = dict(
            zip([get_full_iri(i) for i in self.kb.thing.instances], list(range(len(self.kb.thing.instances)))))
        """
        self.dummy_pos = 'DUMMY_POS'
        self.dummy_neg = 'DUMMY_NEG'

        ## initialize search tree
        self.root = self.rho.getNode(self.kb.thing, root=True)

    def generate_concepts(self, max_length_of_concept, num_of_concepts_refined, min_size_of_concept):

        refined_nodes = set()

        explored_nodes = {self.rho.getNode(i, parent_node=self.root) for i in
                          self.rho.refine(self.root, maxlength=max_length_of_concept)}
        refined_nodes.add(self.root)

        while len(refined_nodes) < num_of_concepts_refined:
            try:
                next_node = explored_nodes.pop()  # Theoretically it should not be
            except:
                print('Explored nodes exhausted.')
                break

            assert isinstance(next_node, Node)
            assert isinstance(next_node.concept, Concept)

            if next_node in refined_nodes:
                continue
            refinements = (self.rho.getNode(i, parent_node=next_node) for i in self.rho.refine(next_node,maxlength=max_length_of_concept))

            for child in refinements:
                if len(child.concept.instances) >= min_size_of_concept:
                    next_node.add_children(child)
                    explored_nodes.add(child)

            refined_nodes.add(next_node)

        concepts = list(i.concept for i in refined_nodes.union(explored_nodes)
                        if len(i.concept.instances) >= min_size_of_concept)

        # VERY INEFFICIENT
        indvs = set()
        for i in concepts:
            indvs.update(i.instances)
        self.indx = dict(
            zip([get_full_iri(i) for i in indvs], list(range(len(indvs)))))

        self.indx[self.dummy_pos] = len(self.indx)
        self.indx[self.dummy_neg] = len(self.indx)
        del indvs

        with open(self.storage_path + '/index.json', 'w') as file_descriptor:
            json.dump(self.indx, file_descriptor, sort_keys=True, indent=3)

        self.logger.info('Number of concepts refined: {0}'.format(len(refined_nodes)))
        self.logger.info('Number of valid concepts generated: {0}'.format(len(concepts)))
        return concepts

    def pos_neg_sampling_from_concept(self, c, number):

        if len(c.instances) > number // 2:
            sampled_positives = random.sample(c.instances, number // 2)
        else:
            sampled_positives = c.instances
        negatives = list(self.kb.thing.instances - c.instances)
        diff = len(sampled_positives) - len(negatives)
        if diff > 0:
            for i in range(diff):
                negatives.append(self.dummy_neg)
        else:
            negatives = random.sample(self.kb.thing.instances - c.instances,
                                      number // 2)

        return sampled_positives, negatives

    def convert_data(self, concepts, labels, params):
        """

        :param concepts:
        :param labels:
        :param params:
        :return:
        """

        target_concept, X, y = [], [], []
        self.logger.info('Training data is being generated.')
        # Generate Training Data
        for _ in range(params['num_of_times_sample_per_concept']):
            for c in concepts:
                pos, neg = self.pos_neg_sampling_from_concept(c, params['num_of_inputs_for_model'])

                vec_of_f_scores = score_with_labels(pos=pos, neg=neg, labels=labels)

                y.append(vec_of_f_scores)

                idx_pos = [self.indx[get_full_iri(i)] if i != self.dummy_pos else self.indx[i] for i in
                           list(pos) + [self.dummy_pos for _ in
                                        range(params['num_of_inputs_for_model'] // 2 - len(pos))]]

                idx_neg = [self.indx[get_full_iri(i)] if i != self.dummy_neg else self.indx[i] for i in
                           list(neg) + [self.dummy_neg for _ in
                                        range(params['num_of_inputs_for_model'] // 2 - len(neg))]]

                input_ = idx_pos + idx_neg
                assert len(input_) == params['num_of_inputs_for_model']
                X.append(input_)

                target_concept.append(c.str)
        return target_concept, X, y

    def score_with_instances(self, *, pos, neg, instances):

        return self.quality_func.score(pos=pos, neg=neg, instances=instances)

    def save(self, concepts: Iterable[Concept], path='Testing.json', sample_size_for_pos_neg=None):

        test_data = {}

        X_pos = []
        X_neg = []
        Target = []
        for c in concepts:
            target = c.str
            pos, neg = self.pos_neg_sampling_from_concept(c, number=sample_size_for_pos_neg)
            pos = list(pos)
            cleaned_neg = list(i for i in neg if i != self.dummy_neg)
            if len(cleaned_neg) == 0:
                continue

            X_pos.append(pos)
            X_neg.append(cleaned_neg)
            Target.append(target)
            test_data[str(len(test_data)) + '.th learning problem'] = {'Target': target,
                                                                       'Positives': [get_full_iri(i) for i in pos],
                                                                       'Negatives': [get_full_iri(i) for i in
                                                                                     cleaned_neg]}

        self.logger.info('Number of concepts at testing {0}'.format(len(test_data)))
        with open(path, 'w') as jsonfile:
            json.dump(test_data, jsonfile, indent=4)
        return Target, X_pos, X_neg


class DummyQuality(AbstractScorer):
    def __init__(self):
        super().__init__({}, {}, {})

    def apply(self, n):
        if isinstance(n, Set):
            for i in n:
                i.quality = 0.5
        elif isinstance(n, Node):
            n.quality = 0.5
        else:
            raise ValueError


class DummyHeuristic(AbstractScorer):
    def __init__(self):
        super().__init__({}, {}, {})

    def apply(self, n):
        if isinstance(n, Set):
            for i in n:
                i.heuristic = 0.5
        elif isinstance(n, Node):
            n.heuristic = 0.5
        else:
            raise ValueError


class DLLearnerBinder:
    def __init__(self, path='/home/demir/Desktop/DL/dllearner-1.4.0/'):
        assert path
        self.execute_dl_learner_path = path

    def generate_config(self, knowledge_base_path, algorithm, positives, negatives, config_path,
                        num_of_concepts_tested):

        Text = list()
        pos_string = "{ "
        neg_string = "{ "
        for i in positives:
            pos_string += "\"" + str(
                i) + "\","
        for j in negatives:
            neg_string += "\"" + str(
                j) + "\","

        pos_string = pos_string[:-1]
        pos_string += "}"

        neg_string = neg_string[:-1]
        neg_string += "}"

        Text.append("rendering = \"dlsyntax\"")
        Text.append("// knowledge source definition")

        # perform cross validation
        Text.append("cli.type = \"org.dllearner.cli.CLI\"")
        # Text.append("cli.performCrossValidation = \"true\"")
        # Text.append("cli.nrOfFolds = 10\n")

        Text.append("ks.type = \"OWL File\"")
        Text.append("\n")

        Text.append("// knowledge source definition")

        Text.append(
            "ks.fileName = \"" + knowledge_base_path + '\"')
        # Text.append(
        #    "ks.fileName = \"" + '/home/demir/Desktop/DL/dllearner-1.4.0/examples/carcinogenesis/carcinogenesis.owl\"')  # carcinogenesis/carcinogenesis.ow

        Text.append("\n")

        Text.append("reasoner.type = \"closed world reasoner\"")
        Text.append("reasoner.sources = { ks }")
        Text.append("\n")

        Text.append("lp.type = \"PosNegLPStandard\"")
        Text.append("accuracyMethod.type = \"fmeasure\"")

        Text.append("\n")

        Text.append("lp.positiveExamples =" + pos_string)
        Text.append("\n")

        Text.append("lp.negativeExamples =" + neg_string)
        Text.append("\n")
        Text.append("alg.writeSearchTree = \"true\"")

        Text.append("op.type = \"rho\"")

        Text.append("op.useCardinalityRestrictions = \"false\"")

        # Text.append(
        #     "alg.searchTreeFile =\"" + config_path + '_search_tree.txt\"')  # carcinogenesis/carcinogenesis.ow

        if algorithm == 'celoe':
            Text.append("alg.type = \"celoe\"")
            Text.append("alg.maxClassExpressionTests = " + str(num_of_concepts_tested))
            Text.append("alg.stopOnFirstDefinition = \"true\"")
        elif algorithm == 'ocel':
            Text.append("alg.type = \"ocel\"")
            Text.append("alg.maxClassDescriptionTests = " + str(num_of_concepts_tested))
            Text.append("alg.showBenchmarkInformation = \"true\"")
        elif algorithm == 'eltl':
            Text.append("alg.type = \"eltl\"")
            Text.append("alg.maxNrOfResults = \"1\"")
            Text.append("alg.stopOnFirstDefinition = \"true\"")
        else:
            raise ValueError('Wrong algorithm choosen.')
        Text.append("\n")

        pathToConfig = config_path + '.conf'  # /home/demir/Desktop/DL/DL-Learner-1.3.0/examples/family-benchmark

        file = open(pathToConfig, "wb")

        for i in Text:
            file.write(i.encode("utf-8"))
            file.write("\n".encode("utf-8"))
        file.close()
        return pathToConfig

    def parse_output(self, results, config_path, serialize):
        # output_of_dl.append('### ' + pathToConfig + ' ends ###')

        top_predictions = None
        for ith, lines in enumerate(results):
            if 'solutions:' in lines:
                top_predictions = results[ith:]

        # top_predictions must have the following form
        """solutions:
        1: Parent(pred.acc.: 100.00 %, F - measure: 100.00 %)
        2: ⊤ (pred.acc.: 50.00 %, F-measure: 66.67 %)
        3: Person(pred.acc.: 50.00 %, F - measure: 66.67 %)
        """
        try:
            assert 'solutions:' in top_predictions[0]
        except:
            print(top_predictions)
            print('PARSING ERROR')
            exit(1)
        str_f_measure = 'F-measure: '
        assert '1: ' in top_predictions[1]
        assert 'pred. acc.:' in top_predictions[1]
        assert str_f_measure in top_predictions[1]

        # Get last numerical value from first item
        best_pred_info = top_predictions[1]

        best_pred = best_pred_info[best_pred_info.index('1: ') + 3:best_pred_info.index(' (pred. acc.:')]

        f_measure = best_pred_info[best_pred_info.index(str_f_measure) + len(str_f_measure): -1]
        assert f_measure[-1] == '%'
        f_measure = float(f_measure[:-1])

        if serialize:
            f_name = config_path + '_' + 'Result.txt'
            with open(f_name, 'w') as handle:
                for sentence in results:
                    handle.write(sentence + '\n')
            handle.close()

        return best_pred, f_measure

    def pipeline(self, *, knowledge_base_path, algorithm, positives, negatives,
                 path_name, num_of_concepts_tested,
                 expand_goal_node_furhter=False,
                 name_of_Example=None, show_path=False):
        if algorithm is None:
            raise ValueError

        print('####### ', algorithm, ' starts ####### ')

        config_path = path_name + '_' + algorithm + '_' + str(num_of_concepts_tested)

        pathToConfig = self.generate_config(knowledge_base_path=knowledge_base_path,
                                            algorithm=algorithm,
                                            positives=positives, negatives=negatives,
                                            config_path=config_path, num_of_concepts_tested=num_of_concepts_tested)

        output_of_dl = list()

        output_of_dl.append('\n\n')
        output_of_dl.append('### ' + pathToConfig + ' starts ###')

        result = subprocess.run([self.execute_dl_learner_path + 'bin/cli', pathToConfig], stdout=subprocess.PIPE,
                                universal_newlines=True)

        lines = result.stdout.splitlines()
        output_of_dl.extend(lines)

        return self.parse_output(output_of_dl, config_path=config_path, serialize=False)
