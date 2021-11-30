path_knowledge_base="$PWD/KGs/Biopax/biopax.owl"
path_of_experiment_folder="$PWD/Trained_25dim/NeroBiopax"
path_of_json_learning_problems="$PWD/LPs/Biopax/lp.json"
echo "##################"
echo "Evaluation Starts on DRILL Biopax benchmark learning problems"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems"
echo "Evaluation Ends"

path_knowledge_base="$PWD/KGs/Mutagenesis/mutagenesis.owl"
path_of_experiment_folder="$PWD/Trained_25dim/NeroMutagenesis"
path_of_json_learning_problems="$PWD/LPs/Mutagenesis/lp_dl_learner.json"
echo "##################"
echo "Evaluation Starts on DL-Learner Mutagenesis benchmark learning problems"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems"
echo "Evaluation Ends"


path_of_json_learning_problems="$PWD/LPs/Mutagenesis/lp.json"
echo "##################"
echo "Evaluation Starts on DL-Learner Mutagenesis benchmark learning problems"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems"
echo "Evaluation Ends"


exit 1

# Path of an experiment folder
path_knowledge_base="$PWD/KGs/Family/family-benchmark_rich_background.owl"
path_of_json_learning_problems="$PWD/LPs/Family/lp_dl_learner.json"
path_of_experiment_folder="$PWD/Trained/NeroFamily"
echo "##################"
echo "Evaluation Starts on DL-Learner Family benchmark learning problems"
echo "KB Path: $path_knowledge_base"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems"
echo "Evaluation Ends"


path_of_experiment_folder="$PWD/Trained/NeroFamily"
path_of_json_learning_problems="$PWD/LPs/Family/lp.json"
echo "##################"
echo "Evaluation Starts on DRILL Family benchmark learning problems"
echo "KB Path: $path_knowledge_base"
python reproduce_experiments.py --path_of_experiment_folder "$path_of_experiment_folder" --path_knowledge_base "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems"
echo "Evaluation Ends"




# Select Datasets and Embeddings
path_knowledge_base="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/KGs/Carcinogenesis/carcinogenesis.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions $number_of_target_expressions --num_epochs $num_epochs
echo "Training Ends"


# Select Datasets and Embeddings
path_knowledge_base="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/KGs/Biopax/biopax.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : $num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions $number_of_target_expressions --num_epochs $num_epochs
echo "Training Ends"


# Select Datasets and Embeddings
path_knowledge_base="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/KGs/Lymphography/lymphography.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions 10 --num_epochs 1
echo "Training Ends"