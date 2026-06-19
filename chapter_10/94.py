import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


MODEL_NAME = "openai-community/gpt2-medium"

QUESTION = "What do you call a sweet eaten after dinner?"
MAX_NEW_TOKENS = 40


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2にはpad_tokenがないため、EOSをpad_tokenとして使う
    tokenizer.pad_token = tokenizer.eos_token

    # GPT2-mediumはチャット専用モデルではないため、
    # 簡単なチャットテンプレートを自分で定義する
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "User: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "Assistant: {{ message['content'] }}\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "Assistant:"
        "{% endif %}"
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    messages = [
        {
            "role": "user",
            "content": QUESTION,
        }
    ]

    # チャットテンプレートを適用
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("\n言語モデルに与えるプロンプト:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n生成された応答:")
    print("=" * 80)
    print(response)
    print("=" * 80)


if __name__ == "__main__":
    main()

#出力結果
'''
使用デバイス: cuda
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█| 292/292 [00:00<00:00

言語モデルに与えるプロンプト:
================================================================================
User: What do you call a sweet eaten after dinner?
Assistant:
================================================================================

生成された応答:
================================================================================
 A sweet eaten after dinner.
Assistant: A sweet eaten after dinner.
Assistant: A sweet eaten after dinner.
Assistant: A sweet eaten after dinner.
Assistant: A sweet eaten after
================================================================================
'''