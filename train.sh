# Select Datasets and Embeddings
path_knowledge_base="$PWD/KGs/Family/family-benchmark_rich_background.owl"
number_of_target_expressions=100
num_epochs=1
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : $num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions $number_of_target_expressions --num_epochs $num_epochs
echo "Training Ends"
exit
# Select Datasets and Embeddings
path_knowledge_base="$PWD/KGs/Mutagenesis/mutagenesis.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions $number_of_target_expressions --num_epochs $num_epochs
echo "Training Ends"



# Select Datasets and Embeddings
path_knowledge_base="$PWD/KGs/Carcinogenesis/carcinogenesis.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions $number_of_target_expressions --num_epochs $num_epochs
echo "Training Ends"


# Select Datasets and Embeddings
path_knowledge_base="$PWD/KGs/Biopax/biopax.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : $num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions $number_of_target_expressions --num_epochs $num_epochs
echo "Training Ends"


# Select Datasets and Embeddings
path_knowledge_base="$PWD/KGs/Lymphography/lymphography.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions 10 --num_epochs 1
echo "Training Ends"