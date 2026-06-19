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
'''
使用デバイス: cuda
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█| 292/292 [00:00<00:00
================================================================================
Greedy decoding
--------------------------------------------------------------------------------
[1]
全文:
The movie was full of great moments, but the most memorable was when the characters were reunited.

The movie was full of great moments, but the most memorable was when
続き:
 great moments, but the most memorable was when the characters were reunited.

The movie was full of great moments, but the most memorable was when

================================================================================
Beam search: num_beams=5
--------------------------------------------------------------------------------
[1]
全文:
The movie was full of great moments, but it was also full of bad ones.

The movie was full of great moments, but it was also full of bad ones
続き:
 great moments, but it was also full of bad ones.

The movie was full of great moments, but it was also full of bad ones

[2]
全文:
The movie was full of great moments, but it also had a lot of bad ones.

The movie was full of great moments, but it also had a lot of
続き:
 great moments, but it also had a lot of bad ones.

The movie was full of great moments, but it also had a lot of

[3]
全文:
The movie was full of great moments, but it was also full of bad ones.

The movie was full of great moments.

The movie was full of great
続き:
 great moments, but it was also full of bad ones.

The movie was full of great moments.

The movie was full of great

================================================================================
Sampling: temperature=0.7, top_k=50
--------------------------------------------------------------------------------
[1]
全文:
The movie was full of action, but that's something I really liked about it. When I was playing characters who were on the run, it was funto have them on
続き:
 action, but that's something I really liked about it. When I was playing characters who were on the run, it was fun to have them on

[2]
全文:
The movie was full of the usual tropes of a "horror" movie. It was all about the death of alove interest. It was full of the same kind of
続き:
 the usual tropes of a "horror" movie. It wasall about the death of a love interest. It was full of the same kind of

[3]
全文:
The movie was full of action and action sequences, but I was interested in some of the smaller events and the story of the characters andthe journey.

I love the
続き:
 action and action sequences, but I was interested in some of the smaller events and the story of the characters and the journey.

I love the

[4]
全文:
The movie was full of these scenes, and also in some of them, these two different versions of the film. I think we made a very good film."

In
続き:
 these scenes, and also in some of them, these two different versions of the film. I think we made a very good film."

In

[5]
全文:
The movie was full of the usual stuff — a young woman (Sonia Dutta) who can't hold the past, a woman who's been given a new job
続き:
 the usual stuff — a young woman (Sonia Dutta) who can't hold the past, a woman who's been given a new job

================================================================================
Sampling: temperature=1.0, top_k=50
--------------------------------------------------------------------------------
[1]
全文:
The movie was full of plot twists and surprises, including a character called the "Walking Dead Guy."

In the movie, Mr. Robot leads two separate storylines —
続き:
 plot twists and surprises, including a character called the "Walking Dead Guy."

In the movie, Mr. Robot leads two separate storylines —

[2]
全文:
The movie was full of great character moments, but it never got much in the way of originalideas by a major actor (although we still hadBen Affleck, of course
続き:
 great character moments, but it never got much in the way of original ideas by a major actor (although we still had Ben Affleck, of course

[3]
全文:
The movie was full of laughs. One moment we're at a strip club, eating drinks, the next we're in a movie theater looking into an audienceand a car crashes
続き:
 laughs. One moment we're at a strip club, eating drinks, the next we're in a movie theaterlooking into an audience and a car crashes

[4]
全文:
The movie was full of jokes about what was wrong with him. His "pale blonde hair," his "thick voice," that he's not a "lovable loser
続き:
 jokes about what was wrong with him. His "pale blonde hair," his "thick voice," that he's not a "lovable loser

[5]
全文:
The movie was full of them, as was the first film in the series, A.R.G.U.S. (a reference tothe A.R.
続き:
 them, as was the first film in the series, A.R.G.U.S. (a reference to the A.R.

================================================================================
Sampling: temperature=1.5, top_k=50
--------------------------------------------------------------------------------
[1]
全文:
The movie was full of laughs. They had plenty! It doesn't even pretend it wasn't amusing: They could make us laugh in front of our own television — or on
続き:
 laughs. They had plenty! It doesn't even pretend it wasn't amusing: They could make us laugh in front of our own television — or on

[2]
全文:
The movie was full of dialogue; this version isn't.
Now back to our first mystery… As a movie's been made where people watch one script for 12 hour days
続き:
 dialogue; this version isn't.
Now back to our first mystery… As a movie's been made where people watch one script for 12 hour days

[3]
全文:
The movie was full of moments (such as havinghim take her into some jungle hideaway to teach her how) all reminiscent of the late night talk-show sketch and that
続き:
 moments (such as having him take her into some jungle hideaway to teach her how) all reminiscent of the late night talk-show sketch and that

[4]
全文:
The movie was full of the same jokes about our generation having been so easily manipulated, having been made too easily... I was actually happy, because so many of us thought we
続き:
 the same jokes about our generation having been so easily manipulated, having been made too easily... I was actually happy, because so many of us thought we

[5]
全文:
The movie was full of brilliant lines and scenes of suspense; just a bit on the weak side, and not enough of all in-the-making special effects to take the
続き:
 brilliant lines and scenes of suspense; justa bit on the weak side, and not enough of allin-the-making special effects to take the

================================================================================
Top-p sampling: temperature=1.0, top_p=0.9
--------------------------------------------------------------------------------
[1]
全文:
The movie was full of amazing moments, but the most memorable moment was that scene with the car, which was hilarious. We tried to make this movie a little darker and not
続き:
 amazing moments, but the most memorable moment was that scene with the car, which was hilarious. We tried to make this movie a little darker and not

[2]
全文:
The movie was full of the usual weird and wacky nonsense, which is a thing to do with the TV show. The episode's only other real quirk was the way
続き:
 the usual weird and wacky nonsense, which isa thing to do with the TV show. The episode'sonly other real quirk was the way

[3]
全文:
The movie was full of amazing characters and characters we would have loved to be friends with. I was just hoping that it would help people understand us a little better and maybe let
続き:
 amazing characters and characters we would have loved to be friends with. I was just hoping that it would help people understand us a little better and maybe let

[4]
全文:
The movie was full of "horror" and also contained references to the recent attacks in Parisby terrorists, such as the "brutal slaughter"of innocent children,
続き:
 "horror" and also contained references to the recent attacks in Paris by terrorists, such as the "brutal slaughter" of innocent children,

[5]
全文:
The movie was full of moments of heartbreak. It featured a man who had been shot and survived, but not for long, before he was shot again. This time the
続き:
 moments of heartbreak. It featured a man whohad been shot and survived, but not for long,before he was shot again. This time the

'''
