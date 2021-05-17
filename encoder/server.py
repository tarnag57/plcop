import preprocess
import predict
import server_params
import sys
import socket
import signal
import time
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_tokenizer(filename):
    json_file = open(filename, 'r')
    json_string = json_file.read()
    return tf.keras.preprocessing.text.tokenizer_from_json(json_string)


def run_inference(interpreter, input_data, vocab_size):

    # Prepare the model
    seq_len = input_data.shape[1]
    interpreter.resize_tensor_input(0, [1, seq_len, vocab_size], strict=True)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]


def log_request(data, filename):
    with open(filename, 'a') as f:
        f.write(f"{data}\n")
        f.flush()


def write_log(args, log_data):
    with open(args.log_file, 'a') as f:
        for key in log_data:
            f.write(f"{key}: {log_data[key]}\n")


def generate_handler(server_socket, args, log_data):
    def handler(sig, frame):
        print("Handler received")
        print("Handler received", file=sys.stderr)
        server_socket.close()
        write_log(args, log_data)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)
    return handler


def main():
    print("Spinning up server")
    print("Spinning up server", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    parser = server_params.build_parser()
    args = parser.parse_args()

    print(f"Port: {args.port}", file=sys.stderr)
    tokenizer = load_tokenizer(args.lang)
    vocab_size = len(tokenizer.word_index) + 1

    interpreter = tf.lite.Interpreter(model_path=args.model)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.bind((args.addr, args.port))
    except Exception as e:
        print(f"Error while binding to port {args.port}", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    log_data = {'total_reqs': 0, 'total_embedding_time': 0.,
                'total_handling_time': 0.}

    signal.signal(signal.SIGINT, generate_handler(
        server_socket, args, log_data))

    while (True):
        server_socket.listen()
        conn, addr = server_socket.accept()

        with conn:
            # Receive clause
            data = conn.recv(4096)
            recieve_start = time.perf_counter()
            data = data.decode('ascii')

            if args.log_requests is not None:
                log_request(data, args.log_requests)

            encoded = None
            if args.empty_response is None:
                # Run prediction on the received clause
                preprocess_start = time.perf_counter()
                embedding = predict.preprocess_clause(
                    data, tokenizer, vocab_size, numbered=False)
                encoding_start = time.perf_counter()
                encoded = run_inference(interpreter, embedding, vocab_size)
                encoding_end = time.perf_counter()
                # print(
                #     f"Preprocess time: {encoding_start - preprocess_start}")
                # print(f"Encoding time: {encoding_end - encoding_start}")
                log_data['total_reqs'] += 1
                log_data['total_embedding_time'] += encoding_end - \
                    preprocess_start
            else:
                encoded = [0] * args.empty_response

            # Reply with result
            message = ",".join(map(str, encoded)) + "\n"
            conn.send(message.encode('ascii'))
            send_end = time.perf_counter()
            log_data['total_handling_time'] += send_end - recieve_start
            # print(f"Total time it took: {send_end - recieve_start}")


if __name__ == "__main__":
    main()
