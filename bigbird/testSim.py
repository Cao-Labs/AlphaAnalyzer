import csv
from collections import defaultdict

class GeneOntologyTree:
    def __init__(self, pathOfGOTree, TestMode=1):
        self.GOParent = defaultdict(list)
        self.GOSpace = dict()       # store GO id and namespace of this GO
        self.MFroot = "GO:0003674"
        self.BProot = "GO:0008150"
        self.CCroot = "GO:0005575"
        self.TestMode = TestMode
        self.PrintMessage("Now loading a GO tree from " + str(pathOfGOTree))
        self._loadTree(pathOfGOTree)

    def PrintMessage(self, Mess):       # when you are testing mode, you can get these messages
        if self.TestMode == 1:
            print(Mess)

    def _loadTree(self, pathOfGOTree):
        with open(pathOfGOTree, 'r') as file:
            current_id = None
            for line in file:
                line = line.strip()
                if line.startswith("id: "):
                    current_id = line[4:]
                elif line.startswith("namespace: "):
                    self.GOSpace[current_id] = line[11:]
                elif line.startswith("is_a: "):
                    parent_id = line[6:16]
                    self.GOParent[current_id].append(parent_id)
                    if parent_id not in self.GOParent:
                        self.GOParent[parent_id] = []

    def _CalPrecisionRecall(self, NewPre, NewTrue):
        TP = 0
        FP = 0
        FN = 0
        if len(NewPre) == 0 or len(NewTrue) == 0:
            return (0, 0)  # Some of true GO terms are not in the version GO database, so we should not consider these kind of predictions
        for each in NewTrue:
            if each in NewPre:
                TP += 1
            else:
                FN += 1
        for each in NewPre:
            if each not in NewTrue:
                FP += 1
        precision = float(TP) / (TP + FP) if (TP + FP) > 0 else 0
        recall = float(TP) / (TP + FN) if (TP + FN) > 0 else 0
        return (precision, recall)

    def GOSimilarity(self, GO1, GO2):
        self.PrintMessage("Now compute the similarity of two GO terms sets based on GO tree. We may need to DFS to get the path from GO to the root.")
        if GO1 not in self.GOSpace or GO2 not in self.GOSpace:
            return -1

        if self.GOSpace[GO1] != self.GOSpace[GO2]:      # no similarity for two go terms in different tree
            return 0

        Longest2Share = 0
        CommonStep = 0

        Level1 = [GO1]
        Sets1 = {GO1}
        Level2 = [GO2]
        Sets2 = {GO2}

        StartingLevel = None

        while True:
            tag = 0
            for eachGO in Level1:
                if eachGO in Sets2:
                    StartingLevel = Level1
                    tag = 1
            if tag == 1:
                break
            tag = 0
            for eachGO in Level2:
                if eachGO in Sets1:
                    StartingLevel = Level2
                    tag = 1
            if tag == 1:
                break
            Longest2Share += 1
            temLevel = []
            for eachGO in Level1:
                if (eachGO != self.MFroot) and (eachGO != self.BProot) and (eachGO != self.CCroot) and (eachGO in self.GOParent):
                    for tGO in self.GOParent[eachGO]:
                        temLevel.append(tGO)
                        Sets1.add(tGO)
            Level1 = temLevel
            temLevel = []
            for eachGO in Level2:
                if (eachGO != self.MFroot) and (eachGO != self.BProot) and (eachGO != self.CCroot) and (eachGO in self.GOParent):
                    for tGO in self.GOParent[eachGO]:
                        temLevel.append(tGO)
                        Sets2.add(tGO)
            Level2 = temLevel

            if len(Level1) == 0 or len(Level2) == 0:      # This means one of them already at the root, but another is still not, so they only share the root.
                finalSimilarity = 1.0 / (Longest2Share + 1.0)
                return finalSimilarity

        while True:
            CommonStep += 1
            temLevel = []
            for eachGO in StartingLevel:
                if (eachGO != self.MFroot) and (eachGO != self.BProot) and (eachGO != self.CCroot) and (eachGO in self.GOParent):
                    for tGO in self.GOParent[eachGO]:
                        temLevel.append(tGO)
            StartingLevel = temLevel
            if len(StartingLevel) == 0:
                break

        finalSimilarity = float(CommonStep) / (CommonStep + Longest2Share)
        return finalSimilarity

def load_predictions(filepath):
    predictions = defaultdict(set)
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row[0].startswith("AUTHOR") or row[0].startswith("MODEL") or row[0].startswith("KEYWORDS"):
                continue
            protein_id, go_term = row[0], row[1]
            predictions[protein_id].add(go_term)
    return predictions

def load_actual(filepath):
    actual = defaultdict(set)
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row[0].startswith("AUTHOR") or row[0].startswith("MODEL") or row[0].startswith("KEYWORDS"):
                continue
            protein_id, go_term = row[0], row[1]
            actual[protein_id].add(go_term)
    return actual

def calculate_metrics(predictions, actual, go_tree):
    TP = 0
    FP = 0
    FN = 0

    for protein_id in actual:
        actual_go_terms = actual[protein_id]
        predicted_go_terms = predictions.get(protein_id, set())

        actual_propagated = set()
        predicted_propagated = set()

        for go_term in actual_go_terms:
            propagate_to_root(go_tree, go_term, actual_propagated)

        for go_term in predicted_go_terms:
            propagate_to_root(go_tree, go_term, predicted_propagated)

        for go_term in predicted_propagated:
            if go_term in actual_propagated:
                TP += 1
            else:
                FP += 1

        for go_term in actual_propagated:
            if go_term not in predicted_propagated:
                FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def propagate_to_root(go_tree, go_term, propagated_set):
    if go_term in propagated_set:
        return
    propagated_set.add(go_term)
    if go_term in go_tree.GOParent:
        for parent in go_tree.GOParent[go_term]:
            propagate_to_root(go_tree, parent, propagated_set)


def main():
    go_tree_path = '/data/go.obo'
    predictions_path = '/result/model6_predicted.csv'
    actual_path = '/result/model6_actual.csv'

    go_tree = GeneOntologyTree(go_tree_path)
    predictions = load_predictions(predictions_path)
    actual = load_actual(actual_path)

   # precision, recall = calculate_metrics(predictions, actual, go_tree)
    precision, recall, f1 = calculate_metrics(predictions, actual, go_tree)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')


if __name__ == '__main__':
    main()
