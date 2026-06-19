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
'''
使用デバイス: cuda
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█| 292/292 [00:00<00:00

入力文:
The movie was full of

生成された全文:
The movie was full of action, explosions and moments that felt surreal and realistic. It helped me relax

生成された続き:
 action, explosions and moments that felt surreal and realistic. It helped me relax

生成された各トークンの尤度:
step  token_id            token          decoded   probability      log_prob
--------------------------------------------------------------------------------
   1      2223        'Ġaction'        ' action'    0.04873190   -3.02142143
   2        11              ','              ','    0.25725776   -1.35767674
   3     23171    'Ġexplosions'    ' explosions'    0.03645995   -3.31154084
   4       290           'Ġand'           ' and'    0.38942194   -0.94309187
   5      7188       'Ġmoments'       ' moments'    0.00859849   -4.75616837
   6       326          'Ġthat'          ' that'    0.14945510   -1.90075922
   7      2936          'Ġfelt'          ' felt'    0.08122639   -2.51051521
   8     28201       'Ġsurreal'       ' surreal'    0.00459800   -5.38213348
   9       290           'Ġand'           ' and'    0.14912879   -1.90294492
  10     12653     'Ġrealistic'     ' realistic'    0.01853094   -3.98831344
  11        13              '.'              '.'    0.55935109   -0.58097792
  12       632            'ĠIt'            ' It'    0.17057405   -1.76858568
  13      4193        'Ġhelped'        ' helped'    0.00442470   -5.42055225
  14       502            'Ġme'            ' me'    0.15178630   -1.88528168
  15      8960         'Ġrelax'         ' relax'    0.03935381   -3.23516250

生成部分全体の対数尤度:
-41.96512556

生成部分全体の尤度:
5.953568352353e-19
'''