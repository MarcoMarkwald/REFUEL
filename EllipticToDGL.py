import pandas as pd
import csv
import numpy as np

if __name__ == '__main__':
    labeledNodes = []
    unknownNodes = {}
    datafeatures = pd.read_csv("Data//elliptic_txs_features.csv", header=None, index_col=0)
    datanodes = pd.read_csv("Data//elliptic_txs_classes.csv", header=0, sep=',', index_col=0)
    dataedges = pd.read_csv("Data//elliptic_txs_edgelist.csv", header=0, index_col=False)

    # Find all labeled Nodes
    for d in datanodes.index:
        c = datanodes.loc[d].values[0]
        if c == "1" or c == "2":
            labeledNodes.append(d)

    # Write edges to edge_plus.csv
    connectednodes = []
    with open('Data//edge_plus.csv', 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["src_id", "dst_id"])
        for e in dataedges.values:
            if e[0] in labeledNodes:
                if not connectednodes.__contains__(e[1]):
                    connectednodes.append(e[1])
                writer.writerow([str(int(e[0])), str(int(e[1]))])

            elif e[1] in labeledNodes:
                if not connectednodes.__contains__(e[0]):
                    connectednodes.append(e[0])
                writer.writerow([str(int(e[0])), str(int(e[1]))])

    for c in connectednodes:
        labeledNodes.append(c)

    dropnodes = []
    for d in datanodes.index:
        if d not in labeledNodes:
            dropnodes.append(d)

    datafeatures = datafeatures.drop(dropnodes)

    # DGL needs train_mask and test_mask
    train, test = np.split(datafeatures.sample(frac=1, random_state=42), [int(.7*len(datafeatures))])

    # Write Node Data to node_plus.csv
    with open('Data//node_plus.csv', 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["node_id", "label", "train_mask", "test_mask", "feat"])
        for index in datafeatures.index:
            train_mask = False
            test_mask = False

            data = datafeatures.loc[index].values[1:169]

            features = '"' + str(data[0])

            for d in data:
                features = features + "," + str(d)

            features = features + '"'

            if datanodes.loc[index].values[0] != 'unknown':
                cl = int(datanodes.loc[index].values[0]) - 1
                if index in train.index:
                    train_mask = True
                if index in test.index:
                    test_mask = True
            else:
                cl = 2

            row = [str(index), cl, train_mask, test_mask, features]

            writer.writerow(row)
