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
'''
入力文: The movie was full of [MASK].
[MASK] の top-10 予測
------------------------------
 1. fun             確率: 0.107119  文: The movie was full of fun.
 2. surprises       確率: 0.066345  文: The movie was full of surprises.
 3. drama           確率: 0.044684  文: The movie was full of drama.
 4. stars           確率: 0.027217  文: The movie was full of stars.
 5. laughs          確率: 0.025413  文: The movie was full of laughs.
 6. action          確率: 0.019517  文: The movie was full of action.
 7. excitement      確率: 0.019038  文: The movie was full of excitement.
 8. people          確率: 0.018290  文: The movie was full of people.
 9. tension         確率: 0.015031  文: The movie was full of tension.
10. music           確率: 0.014646  文: The movie was full of music.
'''