# REFUEL
Rule Extraction for Imbalanced Neural Node Classification

# Dependencies
python3
deep graph library
pytorch
scipy
sklearn
numpy
pandas

# Dataset
Both data sets used are publicly available. The Cora dataset can be used directly via the DGL library. The Elliptic dataset (https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) must be prepared with the EllipticToDGL.py file using:

python3 EllipticToDGL.py 

Then a meta.yaml file must be added to the folder. This file contains the following:

dataset_name: ./elliptic_dataset/
edge_data:
- file_name: edge_plus.csv
node_data:
- file_name: node_plus.csv

# REFUEL
Now it is possible to run:

python3 main.py

You will get results for the Cora and Elliptic datasets for the parameters stated in the paper.

