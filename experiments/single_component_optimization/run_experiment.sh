# $1 - model
# $2 - dataset
# $3 - mode

# Set DOCKING_BENCHMARK_DIR to directory with docking_benchmark package

mkdir $1_$2_$3
RESULTS_DIR=$PWD/$1_$2_$3
mkdir $RESULTS_DIR/docked_molecules
python3 $DOCKING_BENCHMARK_DIR/scripts/generate_molecules.py $1 -m $3 --dataset $2 -o $RESULTS_DIR/optimized_molecules.csv --n-cpu 8 
python3 $DOCKING_BENCHMARK_DIR/scripts/dock_and_save.py $RESULTS_DIR/optimized_molecules.csv --n-cpu 16 -od $RESULTS_DIR/docked_molecules
python3 $DOCKING_BENCHMARK_DIR/scripts/docked_ligand_score_decomposition.py -d $RESULTS_DIR/docked_molecules -c $RESULTS_DIR/optimized_molecules.csv -o $RESULTS_DIR/decomposed_optimized_molecules.csv 

