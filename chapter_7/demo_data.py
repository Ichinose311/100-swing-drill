"""Hand-written toy sentences for an offline pipeline demonstration."""
import pandas as pd

def demo_frames():
    train = [
        ("good fun movie", 1), ("great enjoyable story", 1),
        ("excellent good acting", 1), ("fun enjoyable film", 1),
        ("bad boring movie", 0), ("awful dull story", 0),
        ("terrible bad acting", 0), ("boring dull film", 0),
    ]
    dev = [("good enjoyable", 1), ("fun great", 1), ("bad dull", 0), ("boring awful", 0)]
    return tuple(pd.DataFrame(rows, columns=["sentence", "label"]) for rows in (train, dev))
