# Permutation-Invariant Embeddings for Learning Description Logic Expressions 
Traditional symbolic models have been successfully applied to learn Description Logic expressions from a background knowledge and examples in a provably sound and complete fashion. 
However, these models often require to explore a large number of expressions to find an adequate one. 
Although applying the redundancy elimination and the expression simplification rules often reduce the number of explored expressions, long runtimes incurred by the exploration still prohibit large scale applications of state-of-the-art models. 

Here, we present a neural permutation-invariant embedding model (NERO) to alleviate the exploration problem. 
NERO maps any variable-length sets of examples to a quality distribution over Description Logic expressions. 
Hence,NERO can accurately prune inadequate numerous expressions without exploring a single expression.
Importantly, the novel architecture of NERO allows cooperation with a state-of-the-art models in learning an adequate expression in a provably sound and complete fashion.
NERO can be applied within state-of-the-art models to accelerate reasoning process. We theoretically showed that \approach can represent any standard quality function. 


# Installation
Create a anaconda virtual environment and install dependencies.
```sh
git clone https://github.com/dice-group/DeepTunnellingForRefinementOperators
# Create anaconda virtual enviroment
conda env create -f environment.yml
# Active virtual enviroment 
conda activate deeptunnel
# Install ontolearn library
wget https://github.com/dice-group/Ontolearn/archive/refs/tags/0.4.0.zip
unzip 0.4.0.zip
cd Ontolearn-0.4.0
python -c 'from setuptools import setup; setup()' develop
python -c "import ontolearn"
cd ..
```
# Datasets and learning problems 
```sh
# Ontologies with assertions
unzip KGs.zip
# Learning Problems {(E^+, E^-)}
unzip LPs.zip
```

# Unsupervised Training
Executing the following script results in training our model on all benchmark datasets with default parameters.
```sh
sh train.sh
```
For each experiment, the following log info is stored.
```sh
2021-11-29 10:04:05,408 - Experimenter - INFO - Knowledge Base being Initialized /../KGs/Lymphography/lymphography.owl
2021-11-29 10:04:05,461 - Experimenter - INFO - Number of individuals: 148
2021-11-29 10:04:05,461 - Experimenter - INFO - Number of named classes / expressions: 49
2021-11-29 10:04:05,461 - Experimenter - INFO - Number of properties / roles : 0
2021-11-29 10:04:05,462 - Experimenter - INFO - Learning Problems being generated
...
2021-11-29 10:04:06,080 - Experimenter - INFO - TrainingRunTime 0.002 minutes
2021-11-29 10:04:06,080 - Experimenter - INFO - Save the loss epoch trajectory
2021-11-29 10:04:06,081 - Experimenter - INFO - Save Weights
2021-11-29 10:04:06,083 - Experimenter - INFO - Training Loop ends
2021-11-29 10:04:06,090 - Experimenter - INFO - Total Runtime of the experiment:0.20418190956115723
```

# Testing
We have provided a test script that facilitates testing a pretrained model on different datasets with different learning problems.
```sh
sh test.sh
```
Expected output: 
```sh
##################
Evaluate NERO on Family benchmark dataset by using learning problems provided in DL-Learner
##################
Loading Nero...
Nero is loaded:5.244850397109985
Iterating over 18 learning problems
#################### 1.LP ####################
Nero learning..
Nero: Prediction: Sister ⊔ (∃ married.Brother)   F1-score: 0.911         Num. explored Expressions: 100
CELOE learning..
CELOE: Prediction: Sister ⊔ (∃ married.Brother)  F1-score:0.911          Num. explored Expressions: 8038
ELTL learning..
ELTL: Prediction: Female         F1-score:0.804  Num. explored Expressions: not reported.
#################### 2.LP ####################
...
ELTL: Prediction: Male   F1-score:0.884  Num. explored Expressions: not reported.
NERO: F-measure:0.984+-0.039    Runtime:0.309+-1.248    NumClassTested:21.056+-37.298
CELOE: F-measure:0.980+-0.053   Runtime:5.015+-3.720    NumClassTested:1457.333+-3270.442
ELTL: F-measure:0.964+-0.091    Runtime:3.559+-0.616    NumClassTested:-1.000+-0.000
Evaluation Ends
```


# Embeddings of Description Logic Expressions
Here, we fit a regression model on 2D embeddings of 1-length expressions. 
Despite the information loss incurred due to PCA, embeddings of 1-length expressions have a distinct structure.
![alt text](core/figures/regplotfamily_plot.png)
# Deployment
To ease using pre-trained model, we provide an API.
```sh
python deploy_demo.py --path_of_experiments "$PWD/Best/NeroFamily"
# Few seconds later, pretrained model is deployed in a local server
Running on local URL:  http://127.0.0.1:7860/
```
![alt text](core/figures/deploy_1.png)
![alt text](core/figures/deploy_2.png)


# Integrate DL-Learner
```
# Download DL-Learner
wget --no-check-certificate --content-disposition https://github.com/SmartDataAnalytics/DL-Learner/releases/download/1.4.0/dllearner-1.4.0.zip
unzip dllearner-1.4.0.zip
# Test the DL-learner framework
dllearner-1.4.0/bin/cli dllearner-1.4.0/examples/father.conf
```

## Acknowledgement 
...
## Contact
...