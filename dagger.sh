# Runs the solver on all the problems in a given directory.
# Finally, it performs training on the output.

PROLOG=swi
PROLOG_PATH=/usr/bin/swipl
#PROLOG_OPTIONS='-nodebug -L120M -G120M -T100M -g '
PROLOG_OPTIONS='-nodebug -L0 -G0 -T0 -g '
PROOF_LAYOUT=connect
PROVER_PATH=.

if [[ $# -ne 5 && $# -ne 6 ]]; then
    echo "Usage: $0 <ini file> <problem dir> <log dir> <corenum> <stage> [eval ini file] [eval problem dir]"
    exit 2
fi

if [ ! -r "$1" ]; then
    echo "Error: File $1 not found" >&2
    exit 2
fi

if [ ! -r "$2" ]; then
    echo "Error: Directory $2 not found" >&2
    exit 2
fi

INI_FILE=$1
PROBLEM_DIR=$2
OUT_DIR=$3
CORENUM=$4
STAGE=$5

SOLVER_TIMEOUT=1000
ERROR_FILE="./error.txt"

mkdir -p "$OUT_DIR/$PROBLEM_DIR" # TODO make this more elegant
mkdir -p "error/stage${STAGE}"
mkdir -p "./encoder/server_log/stage${STAGE}"

if [[ $# -ne 5 ]]; then
    echo "EVALUATION MODE"
    EVAL_INI_FILE=$5
    EVAL_PROBLEM_DIR=$6
    mkdir -p "$OUT_DIR/$EVAL_PROBLEM_DIR" # TODO make this more elegant
fi

START=$(date +%s.%N)

export OMP_NUM_THREADS=1
echo "Building montecarlo tree for each problem in $PROBLEM_DIR"

SCRIPTS=()
COUNTER=0
# SCRIPT="/usr/bin/timeout --preserve-status -k 5 ${SOLVER_TIMEOUT} python montecarlo.py ${INI_FILE} --port_offset {} --encoder_server ./encoder/server.py --problem_file {} > ${OUT_DIR}/{}.out 2> ${ERROR_FILE}"
# echo ${SCRIPT}
FILES=($(find ${PROBLEM_DIR} -type f -name "*.p" | shuf))
# for F in "${FILES[@]}"; do
#     SCRIPTS+=('/usr/bin/timeout --preserve-status -k 5 ${SOLVER_TIMEOUT} python montecarlo.py ${INI_FILE} --port_offset $COUNTER --encoder_server ./encoder/server.py --problem_file $F > ${OUT_DIR}/$F.out 2> ${ERROR_FILE}')
#     let "COUNTER += 1"
# done

solver() {
    INDEX=$1
    F=$2
    SOLVER_TIMEOUT=$3
    INI_FILE=$4
    OUT_DIR=$5
    ERROR_FILE=$6
    STAGE=$7
    let "BASE = 2100*(${STAGE} + 1)"
    ENCODER_PARAMS="--port_offset ${INDEX} --base_port ${BASE} --encoder_server ./encoder/server.py --log_file_base ./encoder/server_log/stage${STAGE}/p-"
    SCRIPT="/usr/bin/timeout --preserve-status -k 5 ${SOLVER_TIMEOUT} python montecarlo.py ${INI_FILE} ${ENCODER_PARAMS} --problem_file ./${F} >${OUT_DIR}/${F}.out 2>error/stage${STAGE}/error-${INDEX}.txt"
    echo ${SCRIPT}
    eval ${SCRIPT}
}
export -f solver

let "LAST_INDEX = ${#FILES[@]} - 1"
INDICIES=($(seq 0 1 $LAST_INDEX))
# echo "${INDICIES[@]}"
parallel -j $CORENUM --no-notice --link solver ::: ${INDICIES[@]} ::: ${FILES[@]} ::: ${SOLVER_TIMEOUT} ::: ${INI_FILE} ::: ${OUT_DIR} ::: ${ERROR_FILE} ::: ${STAGE}

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo "MCTS time: $DIFF sec"
echo "Wating on logs to be written"
sleep 10

echo "FAILURE COUNT:"
grep -rl "FAILURE" $OUT_DIR/$PROBLEM_DIR/*.out | wc -l
echo "SUCCESS COUNT:"
grep -rl "SUCCESS" $OUT_DIR/$PROBLEM_DIR/*.out | wc -l

if [[ $# -ne 5 ]]; then
    EVAL_START=$(date +%s.%N)
    echo "Building montecarlo tree for each problem in $EVAL_PROBLEM_DIR"
    # SCRIPT="/usr/bin/timeout --preserve-status -k 5 600 python montecarlo.py ${EVAL_INI_FILE} --problem_file {} > ${OUT_DIR}/{}.out 2> ${OUT_DIR}/{}.err"
    SCRIPT="/usr/bin/timeout --preserve-status -k 5 1200 python montecarlo.py ${EVAL_INI_FILE} --problem_file {} > ${OUT_DIR}/{}.out 2> /dev/null"
    find ${EVAL_PROBLEM_DIR} -type f -name "*.p" | shuf | parallel -j $CORENUM --no-notice $SCRIPT
    EVAL_END=$(date +%s.%N)
    EVAL_DIFF=$(echo "$EVAL_END - $EVAL_START" | bc)
    echo "EVAL MCTS time: $EVAL_DIFF sec"

    echo "FAILURE COUNT:"
    grep -rl "FAILURE" $OUT_DIR/$EVAL_PROBLEM_DIR/*.out | wc -l
    echo "SUCCESS COUNT:"
    grep -rl "SUCCESS" $OUT_DIR/$EVAL_PROBLEM_DIR/*.out | wc -l
fi

echo "Training xgboost"
export OMP_NUM_THREADS=32
python train_xgboost.py ${INI_FILE} --target_model value >${OUT_DIR}/value_output 2>&1 &
python train_xgboost.py ${INI_FILE} --target_model policy >${OUT_DIR}/policy_output 2>&1 &
wait
export OMP_NUM_THREADS=1

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo "Total iteration time: $DIFF sec"
