"""Generate n-grams while preserving the supplied character or word sequence."""


def generate_n_gram(n, sequence):
    if n < 1:
        raise ValueError("n must be at least 1")
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]
