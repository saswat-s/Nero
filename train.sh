# Determine configurations
number_of_target_expressions=10_000
num_embedding_dim=100
num_epochs=1_000
num_individual_per_example=10
num_of_learning_problems_training=25 # |D|=|T| x num_of_learning_problems_training
val_at_every_epochs=500
num_workers=30
learning_rate=0.01
batch_size=1024

# Select Dataset
path_knowledge_base="$PWD/KGs/Family/Family.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num epochs : $num_epochs"
echo "Embedding dim: $num_embedding_dim"
python start_training.py --path_knowledge_base "$path_knowledge_base" --batch_size $batch_size --learning_rate $learning_rate --number_of_target_expressions $number_of_target_expressions --num_embedding_dim $num_embedding_dim --num_epochs $num_epochs --num_individual_per_example $num_individual_per_example --num_of_learning_problems_training $num_of_learning_problems_training --val_at_every_epochs $val_at_every_epochs --num_workers $num_workers
echo "Training Ends"

# Select Dataset
path_knowledge_base="$PWD/KGs/Mutagenesis/Mutagenesis.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num epochs : $num_epochs"
echo "Embedding dim: $num_embedding_dim"
python start_training.py --path_knowledge_base "$path_knowledge_base" --batch_size $batch_size --learning_rate $learning_rate --number_of_target_expressions $number_of_target_expressions --num_embedding_dim $num_embedding_dim --num_epochs $num_epochs --num_individual_per_example $num_individual_per_example --num_of_learning_problems_training $num_of_learning_problems_training --val_at_every_epochs $val_at_every_epochs --num_workers $num_workers
echo "Training Ends"

# Select Dataset
path_knowledge_base="$PWD/KGs/Carcinogenesis/Carcinogenesis.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num epochs : $num_epochs"
echo "Embedding dim: $num_embedding_dim"
python start_training.py --path_knowledge_base "$path_knowledge_base" --batch_size $batch_size --learning_rate $learning_rate --number_of_target_expressions $number_of_target_expressions --num_embedding_dim $num_embedding_dim --num_epochs $num_epochs --num_individual_per_example $num_individual_per_example --num_of_learning_problems_training $num_of_learning_problems_training --val_at_every_epochs $val_at_every_epochs --num_workers $num_workers
echo "Training Ends"


# Select Dataset
path_knowledge_base="$PWD/KGs/Biopax/Biopax.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num epochs : $num_epochs"
echo "Embedding dim: $num_embedding_dim"
python start_training.py --path_knowledge_base "$path_knowledge_base" --batch_size $batch_size --learning_rate $learning_rate --number_of_target_expressions $number_of_target_expressions --num_embedding_dim $num_embedding_dim --num_epochs $num_epochs --num_individual_per_example $num_individual_per_example --num_of_learning_problems_training $num_of_learning_problems_training --val_at_every_epochs $val_at_every_epochs --num_workers $num_workers
echo "Training Ends"

# Select Dataset
path_knowledge_base="$PWD/KGs/Lymphography/Lymphography.owl"
echo "##################"
echo "Training Starts"
echo "KB Path: $path_knowledge_base"
echo "Num Labels : $number_of_target_expressions"
echo "Num epochs : $num_epochs"
echo "Embedding dim: $num_embedding_dim"
python start_training.py --path_knowledge_base "$path_knowledge_base" --batch_size $batch_size --learning_rate $learning_rate --number_of_target_expressions $number_of_target_expressions --num_embedding_dim $num_embedding_dim --num_epochs $num_epochs --num_individual_per_example $num_individual_per_example --num_of_learning_problems_training $num_of_learning_problems_training --val_at_every_epochs $val_at_every_epochs --num_workers $num_workers
echo "Training Ends"
