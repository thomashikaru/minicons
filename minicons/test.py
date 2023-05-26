from typing import Text
from scorer import IncrementalLMScorer
from nltk.tokenize import word_tokenize


if __name__ == "__main__":
    model = IncrementalLMScorer(
        "/Users/thomasclark/mit/convo-ados/output/candor/finetune/gpt2", "cpu"
    )

    texts = [
        "I'm in Virginia. Virginia? Um, so I think we're technically classified as the South, but I always thought it was like mid atlantic. Doesn't really feel too much like the south. I am over in California. Yeah. North little. Carolina's difference. Yeah. Yeah. What, what time is it there for you guys? It is 803 Okay. It's 11 03 for us. So yeah. Is this your, is this your first time doing one of these? Uh No, this No. is my fourth time, I think. Oh my gosh, This is, this is technically my second, but the last time the other person didn't show up. Yeah. Yeah. I've had some pretty interesting conversations from people on here. The first time was a teacher from north Carolina. Um, Cool. He was a really nice guy, gave me some life advice with his experience. That's That off was, that now. was a really fun one, in my opinion. Um Second one was a, it was a new mom from uh Denver, that's right, Denver yeah and the third one was like an ex veteran from Alaska.",
        "I haven't seen that.",
        "I don't know anything.",
        "I'm a person.",
        "I've got to go.",
        "He wasn't there.",
        "We think about it.",
    ]

    texts = list(map(lambda x: " ".join(word_tokenize(x)), texts))
    texts = ["<|startoftext|>" + txt + "<|endoftext|>" for txt in texts]

    a, b, c = model.get_surprisals_and_entropies(texts)
    # print(a)
    # print(b)
    # print(c)
    for x, y, z in zip(a, b, c):
        print(x)
        print(y)
        print(z)
