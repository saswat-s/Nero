# Select Datasets and Embeddings
path_knowledge_base="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/KGs/Family/family-benchmark_rich_background.owl"
echo "Training Starts"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions 10 --num_epochs 1
echo "Training Ends"


# Select Datasets and Embeddings
path_knowledge_base="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/KGs/Mutagenesis/mutagenesis.owl"
echo "Training Starts"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions 10 --num_epochs 1
echo "Training Ends"


# Select Datasets and Embeddings
path_knowledge_base="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/KGs/Carcinogenesis/carcinogenesis.owl"
echo "Training Starts"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions 10 --num_epochs 1
echo "Training Ends"


# Select Datasets and Embeddings
path_knowledge_base="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/KGs/Biopax/biopax.owl"
echo "Training Starts"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions 10 --num_epochs 1
echo "Training Ends"


# Select Datasets and Embeddings
path_knowledge_base="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/KGs/Lymphography/lymphography.owl"
echo "Training Starts"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions 10 --num_epochs 1
echo "Training Ends"