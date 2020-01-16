# $1 - model
# $2 - dataset
# $3 - custom scoring function

# Set the following environment variables:
#   * DOCKING_BASELINES_DIR - path to docking_baselines package
#   * CUSTOM_SCORING_DIR - directory containing definitions of custom scoring functions for smina

mkdir $1_$2_$3
RESULTS_DIR=$PWD/$1_$2_$3
python3 $DOCKING_BASELINES_DIR/scripts/generate_molecules.py $1 -m minimize --dataset $2 -o $RESULTS_DIR/optimized_molecules.csv --n-cpu 8
mkdir $RESULTS_DIR/real_physics_docked_molecules
python3 $DOCKING_BASELINES_DIR/scripts/dock_and_save.py $RESULTS_DIR/optimized_molecules.csv --n-cpu 16 -od $RESULTS_DIR/real_physics_docked_molecules
mkdir $RESULTS_DIR/custom_physics_docked_molecules
python3 $DOCKING_BASELINES_DIR/scripts/dock_and_save.py $RESULTS_DIR/optimized_molecules.csv --n-cpu 16 -c $CUSTOM_SCORING_DIR/$3 -od $RESULTS_DIR/custom_physics_docked_molecules
python3 $DOCKING_BASELINES_DIR/scripts/docked_ligand_score_decomposition.py -d $RESULTS_DIR/real_physics_docked_molecules -c $RESULTS_DIR/optimized_molecules.csv -o $RESULTS_DIR/decomposed_real_physics_optimized_molecules.csv
python3 $DOCKING_BASELINES_DIR/scripts/docked_ligand_score_decomposition.py -d $RESULTS_DIR/custom_physics_docked_molecules -c $RESULTS_DIR/optimized_molecules.csv -o $RESULTS_DIR/decomposed_custom_physics_optimized_molecules.csv

