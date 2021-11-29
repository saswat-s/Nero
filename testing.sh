# Path of an experiment folder
path_knowledge_base="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/Experiments/2021-11-29 09:46:31.872713"
path_of_json_learning_problems="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/LPs/Family/lp_dl_learner.json"
echo "##################"
echo "Evaluation Starts on DL-Learner Family benchmark learning problems"
echo "KB Path: $path_knowledge_base"
python reproduce_experiments.py --path_of_experiment_folder "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems"
echo "Evaluation Ends"


path_of_json_learning_problems="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/LPs/Family/lp.json"
echo "##################"
echo "Evaluation Starts on DRILL Family benchmark learning problems"
echo "KB Path: $path_knowledge_base"
python reproduce_experiments.py --path_of_experiment_folder "$path_knowledge_base" --path_of_json_learning_problems "$path_of_json_learning_problems"
echo "Evaluation Ends"

exit 1
# Select Datasets and Embeddings
path_knowledge_base="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/KGs/Mutagenesis/mutagenesis.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions $number_of_target_expressions --num_epochs $num_epochs
echo "Training Ends"



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