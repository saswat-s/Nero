# (1) Folder containing pretrained models
folder_name="Experiments" #"PretrainedNero10K"
# (2) Model
dataset_name="Family"
path_of_experiment_folder="$PWD/$folder_name/Nero$dataset_name"
topK=10
# (3) Path of a knowledge base
path_knowledge_base="$PWD/KGs/$dataset_name/$dataset_name.owl"
# (4) Path of learning problems
path_lp_folder="/LPs/Random_LPs/$dataset_name"
path_of_json_learning_problems_size5="$PWD$path_lp_folder/LP_Random_input_size_5.json"
path_of_json_learning_problems_size10="$PWD$path_lp_folder/LP_Random_input_size_10.json"
path_of_json_learning_problems_size15="$PWD$path_lp_folder/LP_Random_input_size_15.json"
# (5) DL-Learner Binaries
path_dl_learner=$PWD'/dllearner-1.4.0/'

echo "##################"
echo "Evaluate NERO on $path_of_experiment_folder benchmark dataset by using randomly generated learning problems of having 5 individuals"
echo "##################"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems_size5" --path_dl_learner "$path_dl_learner" --topK $topK
echo "Evaluation Ends"

echo "##################"
echo "Evaluate NERO on $path_of_experiment_folder benchmark dataset by using randomly generated learning problems of having 10 individuals"
echo "##################"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems_size10" --path_dl_learner "$path_dl_learner" --topK $topK
echo "Evaluation Ends"


echo "##################"
echo "Evaluate NERO on $path_of_experiment_folder benchmark dataset by using randomly generated learning problems of having 15 individuals"
echo "##################"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems_size15" --path_dl_learner "$path_dl_learner" --topK $topK
echo "Evaluation Ends"

