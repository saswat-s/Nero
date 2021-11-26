
# Select Datasets and Embeddings
path_knowledge_base="/home/demir/Desktop/Softwares/Ontolearn/KGs/Family/family-benchmark_rich_background.owl"
path_lp="/home/demir/Desktop/Softwares/DeepTunnellingForRefinementOperators/LPs/Family/lp_dl_learner.json"

echo "Training Starts"
python start_training.py --path_knowledge_base "$path_knowledge_base" --path_lp "$path_lp"
echo "Training Ends"


#num_episode=100 #denoted by M in the manuscript
#min_num_concepts=3 #denoted by n in the manuscript
#num_of_randomly_created_problems_per_concept=2 # denoted by m in the manuscript
#echo "Training Starts"
#python drill_train.py --path_knowledge_base "$dataset_path" --min_length 3 --num_of_sequential_actions 4 --path_knowledge_base_embeddings "$family_kge" --num_episode $num_episode --min_num_concepts $min_num_concepts --num_of_randomly_created_problems_per_concept $num_of_randomly_created_problems_per_concept
#echo "Training Ends"