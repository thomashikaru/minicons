from typing import Text
from scorer import IncrementalLMScorer
from nltk.tokenize import word_tokenize
from icecream import ic


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


if __name__ == "__main__":

    model = IncrementalLMScorer("gpt2", "cpu", 1)

    texts = [
        "It's 7:00 a.m. and 20,000 loud, angry people were there.",
        "It's 7:00 a.m. and 20,000 loud and angry people were there.",
        "I went to the supermarket to buy a loaf of bread",
        "I went to the supermarket to buy a loaf of cheese",
        "He saw a bird",
        "She saw a bird",
    ]

    # texts = list(map(lambda x: " ".join(word_tokenize(x)), texts))
    # texts = ["<|startoftext|>" + txt + "<|endoftext|>" for txt in texts]

    texts = model.add_special_tokens(texts)

    a, b, c = model.get_surprisals_and_entropies(texts)

    for words, surprisals, entropies in zip(a, b, c):
        print(
            f"\n{bcolors.UNDERLINE}{'word':<20} {'surprisal':>10} {'entropy':>10}{bcolors.ENDC}"
        )
        for word, surprisal, entropy in zip(words, surprisals, entropies):
            print(
                f"{word:<20} {bcolors.OKGREEN}{surprisal:>10.3f} {entropy:>10.3f}{bcolors.ENDC}"
            )
