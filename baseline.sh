DIR="theorems/m2np"
CORES=4

experiments=(
    #    plcop
    #    plcop_save_all_value_1_
    # plcop_remove_duplicates
    # plcop_return_to_root
    plcop_paramodulation
)

for experiment in "${experiments[@]}"; do
    echo "Experiment $experiment"

    echo "STAGE 0"
    CUDA_VISIBLE_DEVICES=-1 bash dagger.sh ini/${experiment}0.ini $DIR out/${experiment}/stage0 $CORES
    exit

    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        echo "STAGE $i"
        CUDA_VISIBLE_DEVICES=-1 bash dagger.sh ini/${experiment}1.ini $DIR out/${experiment}/stage${i} $CORES
    done

done
