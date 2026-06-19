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
