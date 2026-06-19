import math

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "openai-community/gpt2-medium"

SENTENCES = [
    "The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises",
]


def calc_perplexity(model, tokenizer, device, text):
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = math.exp(loss.item())

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return loss.item(), perplexity, tokens


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    print("\nパープレキシティ測定結果")
    print("=" * 80)

    results = []

    for text in SENTENCES:
        loss, ppl, tokens = calc_perplexity(model, tokenizer, device, text)

        results.append({
            "text": text,
            "loss": loss,
            "ppl": ppl,
            "tokens": tokens,
        })

        print(f"\n文: {text}")
        print(f"トークン列: {tokens}")
        print(f"loss: {loss:.6f}")
        print(f"perplexity: {ppl:.6f}")

    print("\nまとめ")
    print("=" * 80)
    print(f"{'sentence':45s} {'loss':>12s} {'perplexity':>15s}")
    print("-" * 80)

    for r in results:
        print(f"{r['text']:45s} {r['loss']:12.6f} {r['ppl']:15.6f}")


if __name__ == "__main__":
    main()

#出力結果
