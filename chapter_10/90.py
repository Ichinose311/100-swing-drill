import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "openai-community/gpt2-medium"
PROMPT = "The movie was full of"
TOP_K = 10


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # プロンプトをトークン化
    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    print("\n入力文:")
    print(PROMPT)

    print("\nプロンプトのトークン化結果:")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    for i, token_id in enumerate(input_ids[0]):
        token = tokens[i]
        decoded = tokenizer.decode([token_id])
        print(f"{i}: id={token_id.item():5d}, token={token!r}, decoded={decoded!r}")

    # 次トークンの確率を計算
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # logits の形状: [batch_size, sequence_length, vocab_size]
    # 最後のトークン位置から、次トークンの分布を取り出す
    next_token_logits = logits[0, -1, :]

    # softmax で確率に変換
    probs = torch.softmax(next_token_logits, dim=-1)

    # 上位10個
    top_probs, top_ids = torch.topk(probs, TOP_K)

    print("\n次に続くトークン上位10個:")
    for rank, (token_id, prob) in enumerate(zip(top_ids, top_probs), start=1):
        token_id = token_id.item()
        prob = prob.item()

        raw_token = tokenizer.convert_ids_to_tokens(token_id)
        decoded_token = tokenizer.decode([token_id])

        print(
            f"{rank:2d}: "
            f"id={token_id:5d}, "
            f"token={raw_token!r}, "
            f"decoded={decoded_token!r}, "
            f"prob={prob:.6f}"
        )


if __name__ == "__main__":
    main()

#出力結果
'''
使用デバイス: cuda
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config.json: 100%|█| 718/718 [00:00<00:00, 1.
tokenizer_config.json: 100%|█| 26.0/26.0 [00:
vocab.json: 100%|█| 1.04M/1.04M [00:00<00:00,
merges.txt: 100%|█| 456k/456k [00:00<00:00, 4
tokenizer.json: 100%|█| 1.36M/1.36M [00:00<00
model.safetensors: 100%|█| 1.52G/1.52G [00:26
Loading weights: 100%|█| 292/292 [00:00<00:00
generation_config.json: 100%|█| 124/124 [00:0

入力文:
The movie was full of

プロンプトのトークン化結果:
0: id=  464, token='The', decoded='The'
1: id= 3807, token='Ġmovie', decoded=' movie'
2: id=  373, token='Ġwas', decoded=' was'
3: id= 1336, token='Ġfull', decoded=' full'
4: id=  286, token='Ġof', decoded=' of'

次に続くトークン上位10個:
 1: id= 1049, token='Ġgreat', decoded=' great', prob=0.023094
 2: id=10288, token='Ġreferences', decoded=' references', prob=0.013512
 3: id= 2223, token='Ġaction', decoded=' action', prob=0.013043
 4: id= 7188, token='Ġmoments', decoded=' moments', prob=0.012449
 5: id=  262, token='Ġthe', decoded=' the', prob=0.011860
 6: id= 3435, token='Ġcharacters', decoded=' characters', prob=0.008720
 7: id=  777, token='Ġthese', decoded=' these', prob=0.007216
 8: id=24072, token='Ġsurprises', decoded=' surprises', prob=0.006894
 9: id= 1257, token='Ġfun', decoded=' fun', prob=0.006526
10: id=  606, token='Ġthem', decoded=' them', prob=0.006154
'''
