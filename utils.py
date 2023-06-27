import torch
import numpy as np
from statistics import median
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


def getClassificationMetrics(metrics, file):
    pre0 = []
    rec0 = []
    f10 = []
    sup0 = []
    pre1 = []
    rec1 = []
    f11 = []
    sup1 = []

    accuracy = []

    premacro = []
    recmacro = []
    f1macro = []
    preavg = []
    recavg = []
    f1avg = []

    for m in metrics:
        pre0.append(m['0']['precision'])
        rec0.append(m['0']['recall'])
        f10.append(m['0']['f1-score'])
        sup0.append(m['0']['support'])
        pre1.append(m['1']['precision'])
        rec1.append(m['1']['recall'])
        f11.append(m['1']['f1-score'])
        sup1.append(m['1']['support'])

        accuracy.append(m['accuracy'])

        premacro.append(m['macro avg']['precision'])
        recmacro.append(m['macro avg']['recall'])
        f1macro.append(m['macro avg']['f1-score'])
        preavg.append(m['weighted avg']['precision'])
        recavg.append(m['weighted avg']['recall'])
        f1avg.append(m['weighted avg']['f1-score'])

    with open(file, "a") as f:
        f.write('Precision (0): ' + str(np.mean(pre0)) + "\n")
        f.write('Recall (0): ' + str(np.mean(rec0)) + "\n")
        f.write('F1-Score (0): ' + str(np.mean(f10)) + "\n")
        f.write('Support (0): ' + str(np.mean(sup0)) + "\n")

        f.write('Precision (1): ' + str(np.mean(pre1)) + "\n")
        f.write('Recall (1): ' + str(np.mean(rec1)) + "\n")
        f.write('F1-Score (1): ' + str(np.mean(f11)) + "\n")
        f.write('Support (1): ' + str(np.mean(sup1)) + "\n")

        f.write('Accuracy: ' + str(np.mean(accuracy)) + "\n")

        f.write('Precision (macro avg): ' + str(np.mean(premacro)) + "\n")
        f.write('Recall (macro avg): ' + str(np.mean(recmacro)) + "\n")
        f.write('F1-Score (macro avg): ' + str(np.mean(f1macro)) + "\n")

        f.write('Precision (weighted avg): ' + str(np.mean(preavg)) + "\n")
        f.write('Recall (weighted avg): ' + str(np.mean(recavg)) + "\n")
        f.write('F1-Score (weighted avg): ' + str(np.mean(f1avg)) + "\n")

    return


def gettreestruct(dot_data):
    string = dot_data.split('\n')
    parent = {}

    for s in string:
        if '->' in s:
            mother = int(s.split(' -> ')[0])
            child = int(s.split(' -> ')[1].split(' ')[0])
            parent[child] = mother

    return parent


def getsplitnodes(parents, impure):
    rules = []

    for i in impure:
        n = i
        while n != 0:
            if n in parents:
                n = parents[n]
                if n not in rules:
                    rules.append(n)
                else:
                    n = 0

    for i in impure:
        if i in rules:
            rules.remove(i)

    return rules


def findrules(nodes, labels, impurethreshold, treedepth, minNumberOfNodes):
    clf = tree.DecisionTreeClassifier(max_depth=treedepth)
    clf = clf.fit(nodes, labels)

    impure = []
    for ind, imp in enumerate(clf.tree_.impurity):
        if imp < impurethreshold:
            if clf.tree_.value[ind][0][0] > clf.tree_.value[ind][0][1] and clf.tree_.value[ind][0][
                0] > minNumberOfNodes:
                impure.append(ind)

    dot_data = tree.export_graphviz(clf, out_file=None)
    parents = gettreestruct(dot_data)

    splitnodes = getsplitnodes(parents, impure)
    decisions = []
    for ind, r in enumerate(splitnodes):
        if r in clf.tree_.children_left:
            decisions.append([clf.tree_.feature[ind + 1], "<=", clf.tree_.threshold[ind + 1]])
        elif r in clf.tree_.children_right:
            decisions.append([clf.tree_.feature[ind + 1], ">=", clf.tree_.threshold[ind + 1]])

    numdecisions = len(decisions)
    print("Number of decisions: " + str(numdecisions))

    return decisions, numdecisions


def getDecisions(tr):
    splitnodes = []
    for ind, imp in enumerate(tr.tree_.impurity):
        splitnodes.append(ind)

    dot_data = tree.export_graphviz(tr, out_file=None)
    parents = gettreestruct(dot_data)

    decisions = []
    for ind, r in enumerate(splitnodes):
        if r in tr.tree_.children_left:
            decisions.append([tr.tree_.feature[parents[r]], "<=", tr.tree_.threshold[parents[r]]])
        elif r in tr.tree_.children_right:
            decisions.append([tr.tree_.feature[parents[r]], ">=", tr.tree_.threshold[parents[r]]])

    return decisions


def allrules(nodes, labels, treedepth, num_trees):
    clf = RandomForestClassifier(max_depth=treedepth, n_estimators=num_trees)
    clf = clf.fit(nodes.to(torch.device("cpu")), labels.to(torch.device("cpu")))

    splitnodes = []
    for tr in clf.estimators_:
        decisions = getDecisions(tr)
        for d in decisions:
            splitnodes.append(d)

    numdecisions = len(splitnodes)
    print("Number of decisions: " + str(numdecisions))

    return splitnodes, numdecisions


def addsplitnodes(f, decisions):
    tensor_zeros = [0] * len(decisions)

    for i, r in enumerate(decisions):
        if f[r[0]] <= r[1]:
            tensor_zeros[i] = 1

    return tensor_zeros


def addpaths(f, rules):
    tensor_zeros = [0] * len(rules)

    for i, rule in enumerate(rules):
        for r in rule:
            if r[1] == "<=":
                if f[r[0]] <= r[2]:
                    tensor_zeros[i] = 1
            else:
                if f[r[0]] > r[2]:
                    tensor_zeros[i] = 1
            if tensor_zeros[i] == 0:
                break

    return tensor_zeros


def rulerefinement(decisions):
    dicdec = {}
    for d in decisions:
        if d[0] in dicdec:
            dicdec[d[0]].append(d[2])
        else:
            dicdec[d[0]] = [d[2]]

    rules = []
    for key in dicdec.keys():
        data = np.array(dicdec[key]).reshape(-1, 1)
        rules.append([key, median(data)[0]])

    print("Number of decisions: " + str(len(rules)))

    return rules


def binarylabels(label, file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    newlabels = torch.zeros(label.shape[0], dtype=torch.long)

    # Generate a random permutation of indices
    perm = torch.randperm(len(torch.unique(label)))

    # Select the first two values from the permuted tensor
    random_values = torch.unique(label)[perm[:3]]

    minorityclasssize = 0

    for index, l in enumerate(label):
        if l in random_values:
            newlabels[index] = 1
            minorityclasssize += 1

    with open(file, "a") as f:
        f.write("There are " + str(minorityclasssize) + " many nodes in the minority class. \n")
        f.write("This is equal to " + str(minorityclasssize / len(label)) + "% of all nodes. \n")

    return newlabels.to(device)