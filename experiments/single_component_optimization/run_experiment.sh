# $1 - model
# $2 - dataset
# $3 - mode
# $4 - protein

DOCKING_BENCHMARK_DIR=/mnt/users/tcieplinski/local/my_docking/docking_benchmark
mkdir $1_$2_$3_$4
RESULTS_DIR=$PWD/$1_$2_$3_$4
mkdir $RESULTS_DIR/docked_molecules
python3 $DOCKING_BENCHMARK_DIR/scripts/generate_molecules.py $1 -m $3 --dataset $2 -o $RESULTS_DIR/optimized_molecules.csv --n-cpu 8 -p $4
python3 $DOCKING_BENCHMARK_DIR/scripts/dock_and_save.py $RESULTS_DIR/optimized_molecules.csv --n-cpu 16 -od $RESULTS_DIR/docked_molecules -p $4
python3 $DOCKING_BENCHMARK_DIR/scripts/docked_ligand_score_decomposition.py -d $RESULTS_DIR/docked_molecules -c $RESULTS_DIR/optimized_molecules.csv -o $RESULTS_DIR/decomposed_optimized_molecules.csv -p $4
