from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    """
    複数の事例をまとめて1つのミニバッチに変換する関数

    処理:
    1. input_ids の長さが長い順に並び替える
    2. 最大長に合わせて 0 で padding する
    3. input_ids と label をテンソルとして返す
    """

    # トークン列の長さが長い順に並び替える
    batch = sorted(
        batch,
        key=lambda example: len(example["input_ids"]),
        reverse=True
    )

    # input_idsだけ取り出す
    input_ids_list = [
        example["input_ids"]
        for example in batch
    ]

    # labelだけ取り出す
    labels = torch.stack([
        example["label"]
        for example in batch
    ])

    # 長さが異なるinput_idsを0でpaddingする
    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=0
    )

    return {
        "input_ids": input_ids,
        "label": labels,
    }


def main():
    base_dir = Path(__file__).parent

    train_path = base_dir / "sst_train_examples.pt"

    # 71.pyで作成した訓練データを読み込む
    train_examples = torch.load(train_path)

    # 冒頭4事例を取り出す
    batch = train_examples[:4]

    print("collate前")
    for example in batch:
        print({
            "text": example["text"],
            "label": example["label"],
            "input_ids": example["input_ids"],
        })

    print()
    print("collate後")

    batch_collated = collate(batch)

    print(batch_collated)


if __name__ == "__main__":
    main()

#出力結果
'''
collate前
{'text': 'hide new secretions from the parental units ', 'label': tensor([0.]), 'input_ids': tensor([ 5785,    66,    18,    12, 15095,  1594])}
{'text': 'contains no wit , only labored gags ', 'label': tensor([0.]), 'input_ids': tensor([ 3475,    87, 15888,    90, 27695, 42637])}
{'text': 'that loves its characters and communicates something rather beautiful about human nature ', 'label': tensor([1.]), 'input_ids': tensor([    4,  5053,    45,  3305, 31647,   348,   904,  2815,    47,  1276,
         1964])}
{'text': 'remains utterly satisfied to remain the same throughout ', 'label': tensor([0.]), 'input_ids': tensor([  987, 14528,  4941,   873,    12,   208,   898])}

collate後
{'input_ids': tensor([[    4,  5053,    45,  3305, 31647,   348,   904,  2815,    47,  1276,
          1964],
        [  987, 14528,  4941,   873,    12,   208,   898,     0,     0,     0,
             0],
        [ 5785,    66,    18,    12, 15095,  1594,     0,     0,     0,     0,
             0],
        [ 3475,    87, 15888,    90, 27695, 42637,     0,     0,     0,     0,
             0]]), 'label': tensor([[1.],
        [0.],
        [0.],
        [0.]])}
'''