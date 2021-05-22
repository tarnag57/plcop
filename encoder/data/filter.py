freq = 5

with open("reduction_large.txt", "r") as in_f:
    with open("training_input_red", "w") as out_f:
        lines = in_f.readlines()
        for i, l in enumerate(lines):
            if i % freq == 0:
                out_f.write(l)
