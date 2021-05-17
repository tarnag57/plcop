# DIR="theorems/overfit"
# DIR="theorems/BushyProblems_small"
# DIR="theorems/BushyProblems"
# DIR="theorems/m2n140"
# DIR="theorems/m2np"
DIR="theorems/test"

rm -rf out/*
rm -rf results/*
rm -rf error/*
rm -rf encoder/server_log/*

echo "STAGE 0"
CUDA_VISIBLE_DEVICES=-1 bash dagger.sh ini/plcop0.ini $DIR out/stage0 3 0

for i in 1 2; do
    # for i in 1 2 3 4 5 6 7 8 9; do
    echo "Waiting for ports to clear..."
    echo "STAGE $i"
    CUDA_VISIBLE_DEVICES=-1 bash dagger.sh ini/plcop1.ini $DIR out/stage${i} 3 ${i}
done
