from typing import Text
from scorer import IncrementalLMScorer

if __name__ == "__main__":
    model = IncrementalLMScorer("gpt2-medium", "cpu")

    texts = [
        "Hello this is a test of the function's functionality.",
        "You'd have to be very very very musically inclined I guess to do improv like that . I mean , like , you don't really need to be like , be a good thing . I knew one guy that like , I think we had to like help him out because like , he was just like doing like this and he was like , just talking well , what kind of things ? All right , can you get you like on rhythm at least all one note . No But I was like , okay . Uh",
    ]

    a, b, c = model.get_surprisals_and_entropies(texts)
    print(a)
    print(b)
    print(c)
    for x, y, z in zip(a, b, c):
        print(x)
        print(y)
        print(z)
