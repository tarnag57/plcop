with open("data.txt", "r") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

filtered = []

for c in content:
    if not c.startswith("["):
        if not c.startswith("e"):
            filtered.append(c)
print(filtered)
