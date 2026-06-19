import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


MODEL_NAME = "openai-community/gpt2-medium"
PROMPT = "The movie was full of"
MAX_NEW_TOKENS = 15


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # 続きのテキストを生成しつつ、各ステップのスコアも返す
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0]
    generated_new_ids = generated_ids[input_len:]

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    continuation = tokenizer.decode(generated_new_ids, skip_special_tokens=True)

    print("\n入力文:")
    print(PROMPT)

    print("\n生成された全文:")
    print(full_text)

    print("\n生成された続き:")
    print(continuation)

    print("\n生成された各トークンの尤度:")
    print(f"{'step':>4}  {'token_id':>8}  {'token':>15}  {'decoded':>15}  {'probability':>12}  {'log_prob':>12}")
    print("-" * 80)

    total_log_prob = 0.0

    # outputs.scores[t] は、t番目に生成されたトークンを予測するときのスコア
    for step, (token_id, score) in enumerate(zip(generated_new_ids, outputs.scores), start=1):
        probs = torch.softmax(score[0], dim=-1)

        token_prob = probs[token_id].item()
        token_log_prob = torch.log(probs[token_id]).item()

        total_log_prob += token_log_prob

        raw_token = tokenizer.convert_ids_to_tokens(token_id.item())
        decoded_token = tokenizer.decode([token_id.item()])

        print(
            f"{step:4d}  "
            f"{token_id.item():8d}  "
            f"{raw_token!r:>15}  "
            f"{decoded_token!r:>15}  "
            f"{token_prob:12.8f}  "
            f"{token_log_prob:12.8f}"
        )

    sequence_prob = torch.exp(torch.tensor(total_log_prob)).item()

    print("\n生成部分全体の対数尤度:")
    print(f"{total_log_prob:.8f}")

    print("\n生成部分全体の尤度:")
    print(f"{sequence_prob:.12e}")


if __name__ == "__main__":
    main()

#出力結果
