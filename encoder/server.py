import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import time
import socket

import server_params
import predict
import preprocess


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


def main():
    parser = server_params.build_parser()
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.lang)
    vocab_size = len(tokenizer.word_index) + 1

    interpreter = tf.lite.Interpreter(model_path=args.model)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.addr, args.port))

        while (True):
            s.listen()
            conn, addr = s.accept()

            with conn:
                # Receive clause
                data = conn.recv(4096)
                recieve_start = time.perf_counter()
                data = data.decode('ascii')
                # print(f"Received: {data}")

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
                    print(
                        f"Preprocess time: {encoding_start - preprocess_start}")
                    print(f"Encoding time: {encoding_end - encoding_start}")
                else:
                    encoded = [0] * args.empty_response

                # Reply with result
                message = ",".join(map(str, encoded)) + "\n"
                conn.send(message.encode('ascii'))
                send_end = time.perf_counter()
                # print(f"Total time it took: {send_end - recieve_start}")


if __name__ == "__main__":
    main()
