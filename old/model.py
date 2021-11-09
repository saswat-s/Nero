import torch
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from ontolearn import BaseConceptLearner, Node, AbstractScorer, DLFOILHeuristic, CELOEHeuristic
from ontolearn.util import get_full_iri
import json
import numpy as np

from typing import List, AnyStr


class DeepT(torch.nn.Module):
    """ Deep tunnelling for Refinement Operators"""

    def __init__(self, params):
        super(DeepT, self).__init__()
        assert params

        self.embedding_dim = params['num_dim']
        self.num_instances = params['num_instances']
        self.num_outputs = params['num_of_outputs']
        self.num_of_inputs_for_model = params['num_of_inputs_for_model']

        self.embedding = torch.nn.Embedding(self.num_instances, self.embedding_dim, padding_idx=0)
        self.bn1_emb = torch.nn.BatchNorm1d(1)

        self.fc1 = torch.nn.Linear(self.embedding_dim * self.num_of_inputs_for_model,
                                   self.embedding_dim * self.num_of_inputs_for_model)
        self.bn1_h1 = torch.nn.BatchNorm1d(self.embedding_dim * self.num_of_inputs_for_model)

        self.fc2 = torch.nn.Linear(self.embedding_dim * self.num_of_inputs_for_model,
                                   self.embedding_dim * self.num_of_inputs_for_model)

        self.bn1_h2 = torch.nn.BatchNorm1d(self.embedding_dim * self.num_of_inputs_for_model)

        self.fc3 = torch.nn.Linear(self.embedding_dim * self.num_of_inputs_for_model, self.num_outputs)
        self.bn1_h3 = torch.nn.BatchNorm1d(self.num_outputs)

        # self.loss = torch.nn.KLDivLoss(reduction='sum')
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.embedding.weight.data)

    def forward(self, idx):
        emb_idx = self.embedding(idx)
        # reshape
        emb_idx = emb_idx.reshape(emb_idx.shape[0], 1, emb_idx.shape[1] * emb_idx.shape[2])

        emb_idx = F.relu(self.fc1(emb_idx))
        emb_idx = self.bn1_emb(emb_idx)
        emb_idx = F.relu(self.fc2(emb_idx))
        # emb_idx = self.bn1_h2(emb_idx)
        emb_idx = self.fc3(emb_idx)
        # emb_idx = self.bn1_h3(emb_idx)

        emb_idx = emb_idx.squeeze()

        # return torch.softmax(emb_idx, dim=1)
        return torch.sigmoid(emb_idx)


class DeepTHeuristic(AbstractScorer):
    def __init__(self, model=None, labels=None, pos=None, neg=None, unlabelled=None):
        super().__init__(pos, neg, unlabelled)

        self.model = model
        self.labels = labels

    def score(self, *args, **kwargs):
        pass

    def apply(self, node, parent_node=None):
        self.applied += 1
        node.heuristic = node.quality


class DeepTConceptLearner(BaseConceptLearner):
    def __init__(self, *, file_path, knowledge_base, num_of_concepts_tested,refinement_operator, search_tree, quality_func, iter_bound,
                 verbose, terminate_on_goal=False,
                 ignored_concepts={}):

        with open(file_path + 'labels.json', 'r') as file_descriptor:
            labels_index = json.load(file_descriptor)
        self.labels = np.zeros(len(labels_index)).tolist()

        for k, concept in knowledge_base.str_to_concept_from_iterable(labels_index.keys()).items():
            assert len(concept.instances) != 0
            self.labels[labels_index[concept.str]] = concept

        self.labels = np.array(self.labels)

        with open(file_path + 'parameters.json', 'r') as file_descriptor:
            self.param = json.load(file_descriptor)
        with open(file_path + 'index.json', 'r') as file_descriptor:
            self.index = json.load(file_descriptor)

        self.model = DeepT(self.param)

        self.model.load_state_dict(torch.load(file_path + 'kb.name_model.pt', map_location=torch.device('cpu')))
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

        super().__init__(knowledge_base=knowledge_base, refinement_operator=refinement_operator,
                         search_tree=search_tree,
                         quality_func=quality_func,
                         heuristic_func=DeepTHeuristic(model=self.model, labels=self.labels),
                         ignored_concepts=ignored_concepts,
                         terminate_on_goal=terminate_on_goal,
                         num_of_concepts_tested=num_of_concepts_tested,
                         iter_bound=iter_bound, verbose=verbose)



    def initialize_root(self):
        root = self.rho.getNode(self.start_class, root=True)
        self.search_tree.add_node(child_node=root)

    def next_node_to_expand(self, step):
        self.search_tree.sort_search_tree_by_decreasing_order(key='heuristic')
        for n in self.search_tree:
            return n

    def apply_rho(self, node: Node):
        assert isinstance(node, Node)

        refinements = (self.rho.getNode(i, parent_node=node)
                       for i in self.rho.refine(node.concept) if i is not None and i.str not in self.concepts_to_ignore)
        return refinements

    def apply_deep_tunnelling(self, pos, neg):
        set_pos, set_neg = set(pos), set(neg)

        # Sanity checking
        assert not ('DUMMY_POS' in pos or 'DUMMY_NEG' in neg)
        assert len(pos) == len(set_pos) and len(neg) == len(set_neg)

        idx_pos = [self.index[i] for i in pos + ['DUMMY_POS' for _ in
                                                 range(self.param['num_of_inputs_for_model'] // 2 - len(pos))]]

        idx_neg = [self.index[i] for i in neg + ['DUMMY_NEG' for _ in
                                                 range(self.param['num_of_inputs_for_model'] // 2 - len(neg))]]

        x = torch.tensor(np.array(idx_pos + idx_neg), dtype=torch.int64)
        x = x.reshape(1, len(x))
        predictions = self.model.forward(x)
        values, indexes = torch.topk(predictions, 50)
        best_pred = self.labels[indexes]

        goal_found = False
        for ith, pred in enumerate(best_pred):
            n = self.rho.getNode(pred, root=True)
            self.quality_func.apply(n)
            if n.quality == 1:
                goal_found = True
            n.heuristic = values[ith]
            self.search_tree._nodes[n] = n

        if self.terminate_on_goal and goal_found:
            self.search_tree.sort_search_tree_by_decreasing_order(key='quality')
            return list(self.search_tree.get_top_n_nodes(1))[0]

    def predict(self, pos: List[AnyStr], neg: List[AnyStr]):
        """
        @param pos
        @param neg:
        @return:
        """

        self.search_tree.set_positive_negative_examples(p=pos, n=neg, all_instances=self.kb.thing.instances)
        self.initialize_root()

        self.apply_deep_tunnelling(pos=pos, neg=neg)

        for j in range(1, self.iter_bound):
            if self.quality_func.applied>=self.num_of_concepts_tested:
                break

            node_to_expand = self.next_node_to_expand(j)

            for ref in self.apply_rho(node_to_expand):
                goal_found = self.search_tree.add_node(parent_node=node_to_expand, child_node=ref)
                if goal_found:
                    if self.verbose:  # TODO write a function for logging and output that integrates verbose.
                        print('Goal found after {0} number of concepts tested.'.format(
                            self.search_tree.expressionTests))
                    if self.terminate_on_goal:
                        self.search_tree.sort_search_tree_by_decreasing_order(key='quality')
                        return list(self.search_tree.get_top_n_nodes(1))[0]

        return list(self.search_tree.get_top_n_nodes(1))[0]
