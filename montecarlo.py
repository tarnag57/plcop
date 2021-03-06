import subprocess
import sys
import signal
import os
import time
# import pyswip
import numpy as np
import xgboost as xgb
from scipy.sparse import dok_matrix

import params
import util

'''
This python script functions as the entry point for each solver run.
It parses the solver parameters and prepares the environment.
Any Python-defined function can be described here and then passed
to plcop using pyswip.

Note: This script is called separately for each problem file.
'''

# load parameters
args = params.getArgs()
# print("\n\n")
# for k in args.keys():
#     print(k, ": ", args[k])
# print("\n\n")

# make sure training data dir exists
value_train_dir = "{}/train_value".format(args.outdir)
policy_train_dir = "{}/train_policy".format(args.outdir)
clause_dir = "{}/clauses".format(args.outdir)
proof_dir = "{}/proofs".format(args.outdir)
os.makedirs(value_train_dir, exist_ok=True)
os.makedirs(policy_train_dir, exist_ok=True)
os.makedirs(clause_dir, exist_ok=True)
os.makedirs(proof_dir, exist_ok=True)

encoder_server = None


def signal_handler(sig, frame):
    if encoder_server is not None:
        print("Shutting down server")
        encoder_server.send_signal(signal.SIGINT)
        time.sleep(10)
    sys.exit(0)


def start_server(args):
    port = args.base_port + args.port_offset
    process = subprocess.Popen(['python', args.encoder_server,
                                '--port', str(port),
                                '--log_file', args.log_file_base + str(port),
                                '--model', args.model_file,
                                '--lang', args.lang_file
                                ])
    print("Created server")
    time.sleep(10)
    return process


signal.signal(signal.SIGTERM, signal_handler)

# We should start the server at all times since the xgboost training requires the
# embedding of states
encoder_server = start_server(args)

# If a trained xgboost model is passed, we should load it
if args.model_type == "xgboost" and args.guided > 0:
    assert args.guidance_dir is not None
    value_modelfile = "{}/value_xgb".format(args.guidance_dir)
    policy_modelfile = "{}/policy_xgb".format(args.guidance_dir)
    if args.guided == 1:  # using python to access xgboost
        value_model = xgb.Booster({'n_job': 4})
        value_model.load_model(value_modelfile)
        policy_model = xgb.Booster({'n_job': 4})
        policy_model.load_model(policy_modelfile)


def value_xgboost(state):
    d_state = xgb.DMatrix(state)
    value = value_model.predict(d_state)[0]
    return value


def policy_xgboost(x):
    d_x = xgb.DMatrix(x)
    scores = policy_model.predict(d_x)
    return scores


def python_value(State, Value):
    n_dim = args.n_dim*4
    sparse = dok_matrix((1, n_dim), dtype=np.float32)
    for (k, v) in State:
        sparse[0, k] = v
    if args.model_type == "xgboost":
        result = value_xgboost(sparse)
    # current pyswip interface cannot pass floats, so we pass an integer
    result = int(result * 1e10)
    Value.unify(result)
    return True


python_value.arity = 2


def python_policy(State, ValidIndices, ActionIndex):
    n_dim = args.n_dim * 5
    sparse = dok_matrix((len(State), n_dim), dtype=np.float32)
    for i, s in enumerate(State):
        for (k, v) in s:
            sparse[i, k] = v
    if args.model_type == "xgboost":
        scores = policy_xgboost(sparse)
    scores = scores[ValidIndices]
    order = np.argsort(scores)
    selected = ValidIndices[order[-1]]
    ActionIndex.unify(selected)
    return True


python_policy.arity = 3

# pyswip.registerForeign(python_value)
# pyswip.registerForeign(python_policy)

leancop_settings = "conj,nodef"
leancop_settings = "{},comp({})".format(leancop_settings, args.pathlim)
leancop_settings = "{},eager_reduction({})".format(
    leancop_settings, args.eager_reduction)
leancop_settings = "{},paramodulation({})".format(
    leancop_settings, args.paramodulation)

Params = "guided({}),cp({}),sim_depth({}),playout_count({}),min_visit_count({}),n_dim({}),playout_time({}), output_format({}), leancop_settings([{}])".format(
    args.guided, args.cp, args.sim_depth, args.playout, args.min_visit_count, args.n_dim, args.playout_time, args.output_format, leancop_settings)
Params = "{},return_to_root({})".format(Params, args.return_to_root)
Params = "{},save_all_proofs({})".format(Params, args.save_all_proofs)
Params = "{},temperature({})".format(Params, args.temperature)
Params = "{},save_all_policy({})".format(Params, args.save_all_policy)
Params = "{},save_all_value({})".format(Params, args.save_all_value)
Params = "{},lemma_features({})".format(Params, args.lemma_features)
Params = "{},inference_limit({})".format(Params, args.inference_limit)
Params = "{},collapse_vars({})".format(Params, args.collapse_vars)
Params = "{},encoder_port({})".format(
    Params, args.base_port + args.port_offset)

if args.guided == 2:  # using c interface, we need to pass the model files as well
    Params = "{},value_modelfile(\"{}\"),policy_modelfile(\"{}\")".format(
        Params, value_modelfile, policy_modelfile)
if args.bigstep_frequency is not None:
    Params = "{},bigstep_frequency({})".format(Params, args.bigstep_frequency)
Params = "[{}]".format(Params)

query = 'mc_run("{}",{},"{}","{}","{}","{}",ExecutionTime)'.format(
    args.problem_file, Params, value_train_dir, policy_train_dir, clause_dir, proof_dir)
# query = 'profile({},[cumulative(true)])'.format(query)
print("Prolog query: \n", query)

# prolog = pyswip.Prolog()
# prolog.consult("montecarlo.pl")
# result = list(prolog.query(query))[0]
# time = result["ExecutionTime"]
# print("Problem {} executed in {} sec".format(args.problem_file, 1.0 * time/1000))

sys.stdout.flush()

full_query = 'swipl -g \'["core/montecarlo.pl"], {}, halt.\''.format(query)
print("Full query: \n", full_query)

subprocess.call(full_query, shell=True)
print("After call")
sys.stdout.flush()

# Shutting down server
if encoder_server is not None:
    encoder_server.send_signal(signal.SIGINT)
    time.sleep(10)
