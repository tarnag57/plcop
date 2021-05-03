import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import socket
import tensorflow as tf

import preprocess
import predict
import server_params


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

    # Running inference
    print("Input details:")
    for i, det in enumerate(input_details):
        print(f"Input {i}: {det}")

    print(input_data.shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]


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
                data = data.decode('ascii')
                print(f"Connection from {addr}, received: {data}")

                # Run prediction on the received clause
                embedding = predict.preprocess_clause(
                    data, tokenizer, vocab_size, numbered=False)
                encoded = run_inference(interpreter, embedding, vocab_size)

                print("Result of encoding:")
                print(encoded)

                # Reply with result
                message = ",".join(map(str, encoded)) + "\n"
                print(message.encode('ascii'))
                conn.send(message.encode('ascii'))


if __name__ == "__main__":
    main()
