import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


MODEL_NAME = "openai-community/gpt2-medium"

QUESTION_1 = "What do you call a sweet eaten after dinner?"
QUESTION_2 = "Please give me the plural form of the word with its spelling in reverse order."

MAX_NEW_TOKENS_1 = 40
MAX_NEW_TOKENS_2 = 60


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2にはpad_tokenがないため、EOSをpad_tokenとして使う
    tokenizer.pad_token = tokenizer.eos_token

    # GPT2-mediumはチャット専用モデルではないため、
    # User / Assistant 形式の簡単なチャットテンプレートを自作する
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

    return tokenizer


def generate_response(model, tokenizer, device, messages, max_new_tokens):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # GPT-2は止まりにくいことがあるため、次の User: や Assistant: が出たら切る
    for stop_word in ["\nUser:", "\nAssistant:"]:
        if stop_word in response:
            response = response.split(stop_word)[0]

    return prompt, response.strip()


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    tokenizer = build_tokenizer()

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # 1ターン目
    messages_1 = [
        {
            "role": "user",
            "content": QUESTION_1,
        }
    ]

    prompt_1, response_1 = generate_response(
        model=model,
        tokenizer=tokenizer,
        device=device,
        messages=messages_1,
        max_new_tokens=MAX_NEW_TOKENS_1,
    )

    print("\n1ターン目のプロンプト:")
    print("=" * 80)
    print(prompt_1)
    print("=" * 80)

    print("\n1ターン目の応答:")
    print("=" * 80)
    print(response_1)
    print("=" * 80)

    # 2ターン目
    # 問題94で生成された応答を assistant 発話として会話履歴に入れる
    messages_2 = [
        {
            "role": "user",
            "content": QUESTION_1,
        },
        {
            "role": "assistant",
            "content": response_1,
        },
        {
            "role": "user",
            "content": QUESTION_2,
        },
    ]

    prompt_2, response_2 = generate_response(
        model=model,
        tokenizer=tokenizer,
        device=device,
        messages=messages_2,
        max_new_tokens=MAX_NEW_TOKENS_2,
    )

    print("\n2ターン目で言語モデルに与えるプロンプト:")
    print("=" * 80)
    print(prompt_2)
    print("=" * 80)

    print("\n2ターン目の応答:")
    print("=" * 80)
    print(response_2)
    print("=" * 80)


if __name__ == "__main__":
    main()

#出力結果
