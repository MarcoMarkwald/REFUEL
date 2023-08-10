import dgl
from dgl.data import CoraFullDataset, FraudYelpDataset

import torch
import torch.nn.functional as F

import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

import models
import utils


def train(g, train_data, test_data, rulevector, num_heads, att_nodes, outdim):
    #train the GAT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = g.ndata['feat']
    labels = g.ndata['label']

    # tensors to Device
    features = features.to(device)
    rulevector = rulevector.to(device)
    labels = labels.to(device)

    model = models.GATModel(g, att_dim=rulevector.size()[1], in_dim=features.size()[1], out_dim=outdim, num_heads=num_heads,
                     att_nodes=att_nodes, hidden_dim=32)

    model = model.to(device)

    # Define the loss function and the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    dur = []
    # Train the model
    for epoch in range(1000):
        if epoch == 0:
            t0 = time.time()

        # Forward pass
        logits = model(features, rulevector)

        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_data], labels[train_data])
        pred = logp.argmax(1)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute classification report on test
        if epoch % 1000 == 0:
            print(classification_report(pred[test_data].to(torch.device("cpu")),
                                        labels[test_data].to(torch.device("cpu"))))
            dur.append(time.time() - t0)
            print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), np.mean(dur)))

    print("Start Classification with RF")
    clf = RandomForestClassifier()
    clf.fit(torch.detach(logp[train_data].to(torch.device("cpu"))), labels[train_data].to(torch.device("cpu")))
    print("Finished calssifcation")

    return classification_report(clf.predict(torch.detach((logp[test_data].to(torch.device("cpu"))))),
                                 labels[test_data].to(torch.device("cpu")), digits=6, output_dict=True, zero_division=1)


def refuelelliptic(num_heads, att_nodes, outdim, treedepth, num_trees):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read Elliptic Dataset
    dataset = dgl.data.CSVDataset('./Data')
    g = dataset[0]
    g = dgl.add_self_loop(g)
    g = g.to(device)

    # set variables
    metrics = []
    arrnumdecisions = []

    features = g.ndata['feat']
    label = g.ndata['label']

    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']

    # Only include labeled nodes in train and test set
    train_test_set = []
    labelednodes = train_mask + test_mask
    for labelindex, l in enumerate(labelednodes):
        if l == 1:
            train_test_set.append(labelindex)

    # kfold with 10 random splits
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(train_test_set)

    occu = 0

    # train method for different folds
    for train_index, test_index in kf.split(train_test_set):
        occu = occu + 1
        print("split no: " + str(occu))
        print("No Features: " + str(len(features[0])))

        train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        for index in train_index:
            train_mask[train_test_set[index]] = True
        for index in test_index:
            test_mask[train_test_set[index]] = True

        # rule extraction
        decisions, numdecisions = utils.allrules(features[train_mask], label[train_mask], treedepth, num_trees)

        arrnumdecisions.append(numdecisions)

        # Rule selection
        refruleset = utils.rulerefinement(decisions)

        # Rulevector calculation
        print("append rulevector")
        rulevector = []
        for index, f in enumerate(features):
            # use only split nodes
            rulevector.append(torch.Tensor(utils.addsplitnodes(f, refruleset)))

        # Graph Attention Network including rules
        metrics.append(train(g, train_mask, test_mask, torch.stack(rulevector), num_heads, att_nodes, outdim))

    utils.getClassificationMetrics(metrics, file)


def refuelcora(file, num_heads, att_nodes, outdim, treedepth, num_trees):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = CoraFullDataset()
    g = data[0]
    g = dgl.add_self_loop(g)
    g = g.to(device)

    metrics = []
    arrnumdecisions = []

    features = g.ndata['feat']
    label = g.ndata['label']

    g.ndata['label'] = utils.binarylabels(label, file)

    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(features)

    for train_index, test_index in kf.split(features):
        train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        for index in train_index:
            train_mask[index] = True
        for index in test_index:
            test_mask[index] = True

        # Rule Extraction
        decisions, numdecisions = utils.allrules(features[train_mask], label[train_mask], treedepth, num_trees)

        arrnumdecisions.append(numdecisions)

        # Rule Selection
        decisions = utils.rulerefinement(decisions)

        # Rulevector calculation
        rulevector = []
        for index, f in enumerate(features):
            # use only split nodes
            rulevector.append(torch.Tensor(utils.addsplitnodes(f, decisions)))

        # Graph Attention Network including rules
        metrics.append(train(g, train_mask, test_mask, torch.stack(rulevector), num_heads, att_nodes, outdim))

    utils.getClassificationMetrics(metrics, file)


def fraudyelprefuel(file, num_heads, att_nodes, outdim, treedepth, num_trees):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data = FraudYelpDataset()
    g = data[0]
    g = g.edge_type_subgraph(['net_rur'])
    g = dgl.add_self_loop(g)
    g = g.to(device)

    metrics = []
    arrnumdecisions = []

    features = g.ndata['feature']
    label = g.ndata['label']

    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(features)

    for train_index, test_index in kf.split(features):
        train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        for index in train_index:
            train_mask[index] = True
        for index in test_index:
            test_mask[index] = True

        print("Find rules")

        decisions, numdecisions = utils.allrules(features[train_mask], label[train_mask], treedepth, num_trees)

        arrnumdecisions.append(numdecisions)

        decisions = utils.rulerefinement(decisions)

        print("Adding rule vectors now")

        rulevector = []
        for index, f in enumerate(features):
            # use only split nodes
            rulevector.append(torch.Tensor(utils.addsplitnodes(f, decisions)))

        # Graph Attention Network including rules
        metrics.append(train(g, train_mask, test_mask, torch.stack(rulevector), num_heads, att_nodes, outdim))

    utils.getClassificationMetrics(metrics, file)


if __name__ == '__main__':
    file = "results.txt"

    numtrees = 100
    numheads = 4

    ellipticatt = 16
    ellipticoutdim = 64
    elliptictreedepth = 5

    with open(file, "a") as f:
        f.write("Num Heads: " + str(numheads) + ", GAT Attention Nodes: " + str(16) + ", Embedding Dim: " + str(
            64) + ", Tree Depth: " + str(5) + ", #Tree: " + str(numtrees) + "\n")

    refuelelliptic(numheads, 16, 64, 5, numtrees)

    with open(file, "a") as f:
        f.write("Num Heads: " + str(numheads) + ", GAT Attention Nodes: " + str(16) + ", Embedding Dim: " + str(
            64) + ", Tree Depth: " + str(5) + ", #Tree: " + str(100) + "\n")

    refuelcora(numheads, 16, 64, 5, numtrees)

    with open(file, "a") as f:
        f.write("Num Heads: " + str(numheads) + ", GAT Attention Nodes: " + str(16) + ", Embedding Dim: " + str(
            64) + ", Tree Depth: " + str(5) + ", #Tree: " + str(100) + "\n")

    fraudyelprefuel(file, numheads, 16, 64, 5, numtrees)
