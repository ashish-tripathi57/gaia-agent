import os


def load_prompt(dir, name):
    with open(os.path.join(dir, f"{name}.txt"), "r", encoding="utf-8") as f:
        return f.read()
