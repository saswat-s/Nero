# Neural Class Expression Learner  [Name will be changed]
In this work, we propose a neural model that accurately tackles the class expression learning problem.
Given a learning problem (a set of positive and a set negative examples), our approach leverages the Set-Transformer algorithm to obtain
permutation invariant continues representations for input examples. Via these permutation invariant representations,
our approach learn a sequence of logical formulae (ALC class expressions) that represent hierarchical explanations. 
Hence, predictions of our model are inherently explainable for the inputs.


# Installation
Create a anaconda virtual environment and install dependencies.
```
git clone https://github.com/dice-group/DeepTunnellingForRefinementOperators
# Create anaconda virtual enviroment
conda env create -f environment.yml
# Active virtual enviroment 
conda activate deeptunnel
```
# Integrate DL-Learner
```
# Download DL-Learner
wget --no-check-certificate --content-disposition https://github.com/SmartDataAnalytics/DL-Learner/releases/download/1.4.0/dllearner-1.4.0.zip
unzip dllearner-1.4.0.zip
# Test the DL-learner framework
dllearner-1.4.0/bin/cli dllearner-1.4.0/examples/father.conf
```
# Preprocessing
Unzip knowledge graphs, embeddings, learning problems and pretrained models.
```
unzip KGs.zip
unzip LPs.zip
```

# Deployment
To ease using pre-trained model, we provide an API.
![alt text](core/deploy_1.png)
![alt text](core/deploy_2.png)


## Acknowledgement 
We based our implementation on the open source implementation of [DRILL](https://arxiv.org/abs/2106.15373).

## Contact
For any further questions or suggestions, please contact:  ```caglar.demir@upb.de```