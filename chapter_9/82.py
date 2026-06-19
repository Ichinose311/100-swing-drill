import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def main():
    model_name = "google-bert/bert-base-uncased"
    text = "The movie was full of [MASK]."

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    model.eval()

    # 文をトークンIDに変換
    inputs = tokenizer(text, return_tensors="pt")

    # [MASK] の位置を取得
    mask_index = torch.where(
        inputs["input_ids"][0] == tokenizer.mask_token_id
    )[0].item()

    # BERTで予測
    with torch.no_grad():
        outputs = model(**inputs)

    # [MASK] 位置の全語彙に対するスコア
    logits = outputs.logits
    mask_logits = logits[0, mask_index]

    # スコアを確率に変換
    probabilities = torch.softmax(mask_logits, dim=0)

    # 確率が高い上位10個を取得
    top_k = 10
    top_probs, top_token_ids = torch.topk(probabilities, top_k)

    print(f"入力文: {text}")
    print(f"[MASK] の top-{top_k} 予測")
    print("-" * 30)

    for rank, (prob, token_id) in enumerate(zip(top_probs, top_token_ids), start=1):
        token = tokenizer.convert_ids_to_tokens(token_id.item())
        completed_text = text.replace("[MASK]", token)

        print(f"{rank:2d}. {token:15s} 確率: {prob.item():.6f}  文: {completed_text}")


if __name__ == "__main__":
    main()

#出力結果
