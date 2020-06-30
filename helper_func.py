import json
import pickle
import random
def serializer(*, object_: object, path: str, serialized_name: str):
    with open(path + '/' + serialized_name + ".p", "wb") as f:
        pickle.dump(object_, f)
    f.close()


def deserializer(*, path: str, serialized_name: str):
    with open(path + "/" + serialized_name + ".p", "rb") as f:
        obj_ = pickle.load(f)
    f.close()
    return obj_


def pipeline_of_dl_learner(self, algorithm, positives, negatives, num_of_concepts_tested, path_name=None,
                           expand_goal_node_furhter=False,
                           name_of_Example=None, show_path=False):

    if algorithm is None:
        raise ValueError

    print('####### ', algorithm, ' starts ####### ')

    execute_dl_learner_path = "/home/demir/Desktop/DL/dllearner-1.4.0/"
    config_path = self.storage_path + '/' + path_name + '_' + algorithm + '_' + str(num_of_concepts_tested)

    def generate_config():
        Text = list()
        pos_string = "{ "
        neg_string = "{ "
        for i in positives:
            pos_string += "\"http://dl-learner.org/carcinogenesis#" + str(
                i) + "\","  # http://www.benchmark.org/family#
        for j in negatives:
            neg_string += "\"http://dl-learner.org/carcinogenesis#" + str(
                j) + "\","  # http://dl-learner.org/carcinogenesis#

        pos_string = pos_string[:-1]
        pos_string += "}"

        neg_string = neg_string[:-1]
        neg_string += "}"

        Text.append("rendering = \"dlsyntax\"")
        Text.append("// knowledge source definition")

        # perform cross validation
        Text.append("cli.type = \"org.dllearner.cli.CLI\"")
        Text.append("cli.performCrossValidation = \"true\"")
        Text.append("cli.nrOfFolds = 10\n")

        Text.append("ks.type = \"OWL File\"")
        Text.append("\n")

        Text.append("// knowledge source definition")
        Text.append(
            "ks.fileName = \"" + '/home/demir/Desktop/DL/dllearner-1.4.0/examples/carcinogenesis/carcinogenesis.owl\"')  # carcinogenesis/carcinogenesis.ow

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


        elif algorithm == 'pceloe':
            Text.append("alg.type = \"pceloe\"")
            # Text.append("alg.maxClassDescriptionTests = 100")
            Text.append("alg.maxClassDescriptionTests = " + str(num_of_concepts_tested))

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

    pathToConfig = generate_config()

    output_of_dl = list()

    output_of_dl.append('\n\n')
    output_of_dl.append('### ' + pathToConfig + ' starts ###')

    return True  # To avoid memeory error
    """
    result = subprocess.run([execute_dl_learner_path + 'bin/cli', pathToConfig], stdout=subprocess.PIPE,
                            universal_newlines=True)

    lines = result.stdout.splitlines()
    output_of_dl.extend(lines)

    # output_of_dl.append('### ' + pathToConfig + ' ends ###')

    f_name = config_path + '_' + 'Result.txt'
    with open(f_name, 'w') as handle:
        for sentence in output_of_dl:
            handle.write(sentence + '\n')
    handle.close()
    return algorithm + ' is Completed.'
    """

def save_as_json(data,path,kb):

    test_data = {}

    for i in data:
        target = i.str
        positives = i.instances
        negatives = kb.thing.instances - i.instances
        try:
            sampled_negatives = random.sample(negatives, len(positives))
        except:
            sampled_negatives=negatives
        test_data[str(len(test_data))+'.th learning problem'] = {'Target': target,
                                     'Positives': list(positives),
                                     'Negatives': list(sampled_negatives)}

    with open(path,'w') as jsonfile:
        json.dump(test_data,jsonfile,indent=4)


def score_with_labels(*, pos, neg, labels):

    pos = set(pos)
    neg = set(neg)

    y = []
    for j in labels:
        individuals = j.instances

        tp = len(pos.intersection(individuals))
        tn = len(neg.difference(individuals))

        fp = len(neg.intersection(individuals))
        fn = len(pos.difference(individuals))
        try:
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f_1 = 2 * ((precision * recall) / (precision + recall))
        except:
            f_1 = 0

        y.append(round(f_1, 5))
    return y
