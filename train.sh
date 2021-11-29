# Select Datasets and Embeddings
path_knowledge_base="$PWD/KGs/Family/family-benchmark_rich_background.owl"
number_of_target_expressions=1000
num_epochs=10
num_individual_per_example=10
num_of_learning_problems_training=2 # |D|=|T| x num_of_learning_problems_training
val_at_every_epochs=5
num_workers=32

echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : $num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions $number_of_target_expressions --num_epochs $num_epochs --num_individual_per_example $num_individual_per_example --num_of_learning_problems_training $num_of_learning_problems_training --val_at_every_epochs $val_at_every_epochs --num_workers $num_workers
echo "Training Ends"


# Select Datasets and Embeddings
path_knowledge_base="$PWD/KGs/Mutagenesis/mutagenesis.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions $number_of_target_expressions --num_epochs $num_epochs --num_individual_per_example $num_individual_per_example --num_of_learning_problems_training $num_of_learning_problems_training --val_at_every_epochs $val_at_every_epochs --num_workers $num_workers
echo "Training Ends"


# Select Datasets and Embeddings
path_knowledge_base="$PWD/KGs/Carcinogenesis/carcinogenesis.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions $number_of_target_expressions --num_epochs $num_epochs --num_individual_per_example $num_individual_per_example --num_of_learning_problems_training $num_of_learning_problems_training --val_at_every_epochs $val_at_every_epochs --num_workers $num_workers
echo "Training Ends"


# Select Datasets and Embeddings
path_knowledge_base="$PWD/KGs/Biopax/biopax.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : $num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions $number_of_target_expressions --num_epochs $num_epochs --num_individual_per_example $num_individual_per_example --num_of_learning_problems_training $num_of_learning_problems_training --val_at_every_epochs $val_at_every_epochs --num_workers $num_workers
echo "Training Ends"

# Select Datasets and Embeddings
path_knowledge_base="$PWD/KGs/Lymphography/lymphography.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num num_epochs : num_epochs"
python start_training.py --path_knowledge_base "$path_knowledge_base" --number_of_target_expressions $number_of_target_expressions --num_epochs $num_epochs --num_individual_per_example $num_individual_per_example --num_of_learning_problems_training $num_of_learning_problems_training --val_at_every_epochs $val_at_every_epochs --num_workers $num_workers
echo "Training Ends"
