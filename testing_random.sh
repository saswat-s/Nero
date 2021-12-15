# (1) Folder containing pretrained models
folder_name="Experiments"
# (2) Model
path_of_experiment_folder="$PWD/$folder_name/2021-12-15 08:30:04.399759"

# (3) Path of a knowledge base
path_knowledge_base="$PWD/KGs/Family/family-benchmark_rich_background.owl"
# (4) Path of learning problems
path_lp_folder='/LPs/Random_LPs/Family'
path_of_json_learning_problems_size5="$PWD$path_lp_folder/LP_Random_input_size_5.json"
path_of_json_learning_problems_size10="$PWD$path_lp_folder/LP_Random_input_size_10.json"
path_of_json_learning_problems_size15="$PWD$path_lp_folder/LP_Random_input_size_15.json"
path_of_json_learning_problems_size25="$PWD$path_lp_folder/LP_Random_input_size_25.json"
path_of_json_learning_problems_size50="$PWD$path_lp_folder/LP_Random_input_size_50.json"
# (5) DL-Learner Binaries
path_dl_learner=$PWD'/dllearner-1.4.0/'

echo "##################"
echo "Evaluate NERO on $path_of_experiment_folder benchmark dataset by using randomly generated learning problems of having 5 individuals"
echo "##################"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems_size5" --path_dl_learner "$path_dl_learner"
echo "Evaluation Ends"
exit 1
echo "##################"
echo "Evaluate NERO on $path_of_experiment_folder benchmark dataset by using randomly generated learning problems of having 10 individuals"
echo "##################"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems_size10" --path_dl_learner "$path_dl_learner"
echo "Evaluation Ends"


echo "##################"
echo "Evaluate NERO on $path_of_experiment_folder benchmark dataset by using randomly generated learning problems of having 15 individuals"
echo "##################"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems_size15" --path_dl_learner "$path_dl_learner"
echo "Evaluation Ends"


echo "##################"
echo "Evaluate NERO on $path_of_experiment_folder benchmark dataset by using randomly generated learning problems of having 25 individuals"
echo "##################"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems_size25" --path_dl_learner "$path_dl_learner"
echo "Evaluation Ends"


echo "##################"
echo "Evaluate NERO on $path_of_experiment_folder benchmark dataset by using randomly generated learning problems of having 50 individuals"
echo "##################"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems_size50" --path_dl_learner "$path_dl_learner"
echo "Evaluation Ends"

