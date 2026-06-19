import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def main():
    model_name = "google-bert/bert-base-uncased"
    text = "The movie was full of [MASK]."

    # トークナイザとBERTのマスク予測モデルを読み込む
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    model.eval()

    # 文をID列に変換
    inputs = tokenizer(text, return_tensors="pt")

    # [MASK] の位置を探す
    mask_index = torch.where(
        inputs["input_ids"][0] == tokenizer.mask_token_id
    )[0].item()

    # BERTで予測
    with torch.no_grad():
        outputs = model(**inputs)

    # [MASK] 位置における全語彙のスコア
    logits = outputs.logits
    mask_logits = logits[0, mask_index]

    # 最もスコアが高いトークンIDを取得
    predicted_token_id = torch.argmax(mask_logits).item()

    # IDをトークンに戻す
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)

    print("予測トークン:", predicted_token)
    print("補完後の文:", text.replace("[MASK]", predicted_token))


if __name__ == "__main__":
    main()

#出力結果
