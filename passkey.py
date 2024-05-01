# There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I
# will quiz you about the important information there. The grass is green. The sky is blue. The sun
# is yellow. Here we go. There and back again. (repeat x times) The pass key is 9054. Remember
# it. 9054 is the pass key. The grass is green. The sky is blue. The sun is yellow. Here we go.
# There and ack again. (repeat y times) What is the pass key? The pass key is

from typing import List, Tuple

import random

_starting_sentence = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
_end_sentence = "What is the pass key? The pass key is: "
_passkey_sentence = "The pass key is {}. Remember it. {} is the pass key."
_repeated_sentence = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
_len = len(_repeated_sentence)
_repeated_sentence = _repeated_sentence * 1000


def generate_passkey_dataset(
    n: int, x: int, lower: int = 0, upper: int = 10000
) -> List[Tuple[str, str]]:
    dataset = []
    for i in range(n):
        label = random.randint(lower, upper)

        text = f"{_starting_sentence} {_repeated_sentence[: _len * x // 2]}{_passkey_sentence.format(label, label)} {_repeated_sentence[: _len * (x - x // 2)]}{_end_sentence}"
        dataset.append((text, label))
    return dataset


dataset = generate_passkey_dataset(3, 100)
for x, y in dataset:
    print(len(x.split()))
    print(y)
    break
