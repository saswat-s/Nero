# (1) Folder containing pretrained models
folder_name="ExperimentsLarge"
path_dl_learner=$PWD'/dllearner-1.4.0/'

# (3) Evaluate NERO on Family benchmark dataset by using learning problems provided in DL-Learner
echo "##################"
echo "Evaluate NERO on Family benchmark dataset by using learning problems provided in DL-Learner"
echo "##################"
# Path of an experiment folder
path_knowledge_base="$PWD/KGs/Family/family-benchmark_rich_background.owl"
path_of_json_learning_problems="$PWD/LPs/Family/lp_dl_learner.json"
path_of_experiment_folder="$PWD/$folder_name/NeroFamily"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems" --path_dl_learner "$path_dl_learner"
echo "Evaluation Ends"
# (5) Evaluate NERO on Mutagenesis benchmark dataset by using learning problems provided in DL-Learner
echo "##################"
echo "Evaluate NERO on Mutagenesis benchmark dataset by using learning problems provided in DL-Learner"
echo "##################"
path_knowledge_base="$PWD/KGs/Mutagenesis/mutagenesis.owl"
path_of_experiment_folder="$PWD/$folder_name/NeroMutagenesis"
path_of_json_learning_problems="$PWD/LPs/Mutagenesis/lp_dl_learner.json"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems"
echo "Evaluation Ends"

# (7) Evaluate NERO on Carcinogenesis benchmark dataset by using learning problems provided in DL-Learner
echo "##################"
echo "Evaluate NERO on Carcinogenesis benchmark dataset by using learning problems provided in DL-Learner"
echo "##################"
path_knowledge_base="$PWD/KGs/Carcinogenesis/carcinogenesis.owl"
path_of_experiment_folder="$PWD/$folder_name/NeroCarcinogenesis"
path_of_json_learning_problems="$PWD/LPs/Carcinogenesis/lp_dl_learner.json"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems"
echo "Evaluation Ends"