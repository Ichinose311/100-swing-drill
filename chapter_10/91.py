import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


MODEL_NAME = "openai-community/gpt2-medium"
PROMPT = "The movie was full of"
MAX_NEW_TOKENS = 30


def generate_and_print(model, tokenizer, device, title, generation_kwargs):
    print("=" * 80)
    print(title)
    print("-" * 80)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )

    for i, output_ids in enumerate(outputs, start=1):
        full_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        # 入力プロンプトの後ろだけを取り出す
        continuation = full_text[len(PROMPT):]

        print(f"[{i}]")
        print("全文:")
        print(full_text)
        print("続き:")
        print(continuation)
        print()


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # 1. Greedy decoding
    # 各時点で最も確率が高いトークンを選ぶ
    generate_and_print(
        model,
        tokenizer,
        device,
        title="Greedy decoding",
        generation_kwargs={
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
        },
    )

    # 2. Beam search
    # 複数の候補列を保持しながら、全体として尤もらしい系列を探す
    generate_and_print(
        model,
        tokenizer,
        device,
        title="Beam search: num_beams=5",
        generation_kwargs={
            "do_sample": False,
            "num_beams": 5,
            "num_return_sequences": 3,
        },
    )

    # 3. Sampling temperature 0.7
    # 低めの temperature なので、比較的安定した生成になりやすい
    generate_and_print(
        model,
        tokenizer,
        device,
        title="Sampling: temperature=0.7, top_k=50",
        generation_kwargs={
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "num_return_sequences": 5,
        },
    )

    # 4. Sampling temperature 1.0
    # 標準的なランダム性
    generate_and_print(
        model,
        tokenizer,
        device,
        title="Sampling: temperature=1.0, top_k=50",
        generation_kwargs={
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
            "num_return_sequences": 5,
        },
    )

    # 5. Sampling temperature 1.5
    # 高めの temperature なので、多様だが不自然な生成も増えやすい
    generate_and_print(
        model,
        tokenizer,
        device,
        title="Sampling: temperature=1.5, top_k=50",
        generation_kwargs={
            "do_sample": True,
            "temperature": 1.5,
            "top_k": 50,
            "num_return_sequences": 5,
        },
    )

    # 6. Top-p sampling
    # 累積確率が top_p になる範囲からサンプリングする
    generate_and_print(
        model,
        tokenizer,
        device,
        title="Top-p sampling: temperature=1.0, top_p=0.9",
        generation_kwargs={
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.9,
            "num_return_sequences": 5,
        },
    )


if __name__ == "__main__":
    main()

#出力結果
