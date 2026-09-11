from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template_string, request


BASE_DIR = Path(__file__).resolve().parent
TASK95_PATH = BASE_DIR / "95_2020ver.py"
DEFAULT_CHECKPOINT = BASE_DIR / "models" / "95_sentencepiece" / "best_model.pt"


PAGE = r"""
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>日英翻訳</title>
  <script src="https://unpkg.com/lucide@0.468.0/dist/umd/lucide.min.js"></script>
  <style>
    :root {
      color-scheme: light;
      --ink: #17201d;
      --muted: #65706c;
      --line: #d8dfdc;
      --paper: #ffffff;
      --wash: #f4f7f5;
      --accent: #087f6b;
      --accent-hover: #066a5a;
      --focus: #e56b4b;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--wash);
      color: var(--ink);
      font-family: Inter, "Noto Sans JP", system-ui, sans-serif;
      letter-spacing: 0;
    }
    header {
      height: 58px;
      border-bottom: 1px solid var(--line);
      background: var(--paper);
      display: flex;
      align-items: center;
    }
    .header-inner, main { width: min(1120px, calc(100% - 32px)); margin: 0 auto; }
    .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; }
    .brand-mark {
      width: 30px; height: 30px; display: grid; place-items: center;
      color: white; background: var(--accent); border-radius: 6px;
    }
    main { padding: 32px 0 48px; }
    .direction {
      display: grid; grid-template-columns: 1fr 42px 1fr;
      align-items: center; gap: 10px; margin-bottom: 12px;
      color: var(--muted); font-size: 14px; font-weight: 650;
    }
    .direction span:last-child { text-align: left; }
    .direction-icon { display: grid; place-items: center; }
    .workspace {
      display: grid; grid-template-columns: 1fr 1fr;
      border: 1px solid var(--line); border-radius: 6px;
      overflow: hidden; background: var(--paper); min-height: 390px;
    }
    .pane { position: relative; min-width: 0; display: flex; flex-direction: column; }
    .pane + .pane { border-left: 1px solid var(--line); background: #fbfcfb; }
    .pane-toolbar {
      height: 48px; flex: 0 0 48px; padding: 0 12px 0 16px;
      border-bottom: 1px solid var(--line); display: flex;
      align-items: center; justify-content: space-between;
      color: var(--muted); font-size: 13px;
    }
    textarea, .result {
      width: 100%; min-height: 292px; flex: 1; border: 0; outline: 0;
      resize: none; padding: 22px; background: transparent; color: var(--ink);
      font: 400 20px/1.75 Inter, "Noto Sans JP", system-ui, sans-serif;
      letter-spacing: 0; overflow-wrap: anywhere; white-space: pre-wrap;
    }
    textarea:focus { box-shadow: inset 0 0 0 2px var(--focus); }
    textarea::placeholder { color: #9aa49f; }
    .result.empty { color: #9aa49f; }
    .actions {
      display: flex; justify-content: flex-end; align-items: center;
      gap: 10px; margin-top: 16px;
    }
    button {
      height: 42px; border: 1px solid transparent; border-radius: 6px;
      display: inline-flex; align-items: center; justify-content: center;
      gap: 8px; cursor: pointer; font: 650 14px/1 system-ui, sans-serif;
      letter-spacing: 0;
    }
    button:focus-visible { outline: 3px solid color-mix(in srgb, var(--focus) 45%, transparent); }
    .primary { min-width: 128px; padding: 0 18px; background: var(--accent); color: white; }
    .primary:hover { background: var(--accent-hover); }
    .primary:disabled { opacity: .55; cursor: wait; }
    .icon-button {
      width: 36px; height: 36px; padding: 0; color: var(--muted);
      background: transparent; border-color: var(--line);
    }
    .icon-button:hover { color: var(--ink); background: var(--wash); }
    .status { min-height: 22px; margin-right: auto; color: var(--muted); font-size: 13px; }
    .status.error { color: var(--danger); }
    .spin { animation: spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (max-width: 720px) {
      .header-inner, main { width: min(100% - 20px, 1120px); }
      main { padding-top: 20px; }
      .direction { grid-template-columns: 1fr 28px 1fr; }
      .workspace { grid-template-columns: 1fr; }
      .pane + .pane { border-left: 0; border-top: 1px solid var(--line); }
      textarea, .result { min-height: 210px; font-size: 18px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div class="brand"><span class="brand-mark"><i data-lucide="languages" size="18"></i></span>日英翻訳</div>
    </div>
  </header>
  <main>
    <div class="direction">
      <span>日本語</span>
      <span class="direction-icon"><i data-lucide="arrow-right" size="18"></i></span>
      <span>English</span>
    </div>
    <form id="translation-form">
      <div class="workspace">
        <section class="pane">
          <div class="pane-toolbar">
            <span id="counter">0 / 1000</span>
            <button class="icon-button" id="clear-button" type="button" title="入力を消去" aria-label="入力を消去"><i data-lucide="x" size="17"></i></button>
          </div>
          <textarea id="source" maxlength="1000" autofocus placeholder="翻訳する日本語を入力" aria-label="翻訳する日本語"></textarea>
        </section>
        <section class="pane">
          <div class="pane-toolbar">
            <span>翻訳結果</span>
            <button class="icon-button" id="copy-button" type="button" title="結果をコピー" aria-label="結果をコピー"><i data-lucide="copy" size="17"></i></button>
          </div>
          <div class="result empty" id="result" aria-live="polite">英訳がここに表示されます</div>
        </section>
      </div>
      <div class="actions">
        <span class="status" id="status"></span>
        <button class="primary" id="submit-button" type="submit"><i data-lucide="languages" size="18"></i><span>翻訳する</span></button>
      </div>
    </form>
  </main>
  <script>
    lucide.createIcons();
    const form = document.getElementById('translation-form');
    const source = document.getElementById('source');
    const result = document.getElementById('result');
    const status = document.getElementById('status');
    const counter = document.getElementById('counter');
    const submit = document.getElementById('submit-button');
    source.addEventListener('input', () => { counter.textContent = `${source.value.length} / 1000`; });
    document.getElementById('clear-button').addEventListener('click', () => {
      source.value = ''; counter.textContent = '0 / 1000'; source.focus();
    });
    document.getElementById('copy-button').addEventListener('click', async () => {
      if (!result.classList.contains('empty')) {
        await navigator.clipboard.writeText(result.textContent);
        status.textContent = 'コピーしました';
      }
    });
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = source.value.trim();
      if (!text) { status.textContent = '日本語を入力してください'; status.className = 'status error'; return; }
      submit.disabled = true;
      submit.innerHTML = '<i data-lucide="loader-circle" class="spin" size="18"></i><span>翻訳中</span>';
      lucide.createIcons(); status.textContent = ''; status.className = 'status';
      try {
        const response = await fetch('/translate', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '翻訳に失敗しました');
        result.textContent = data.translation;
        result.classList.remove('empty');
        status.textContent = `beam ${data.beam_size}`;
      } catch (error) {
        status.textContent = error.message; status.className = 'status error';
      } finally {
        submit.disabled = false;
        submit.innerHTML = '<i data-lucide="languages" size="18"></i><span>翻訳する</span>';
        lucide.createIcons();
      }
    });
  </script>
</body>
</html>
"""


def load_task95():
    spec = importlib.util.spec_from_file_location("task95_server", TASK95_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {TASK95_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_app(checkpoint: Path, beam_size: int) -> Flask:
    task95 = load_task95()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, source_sp, target_sp, saved = task95.load_model_bundle(checkpoint, device)
    max_length = saved["model_config"]["max_length"]
    model_lock = threading.Lock()
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    @app.post("/translate")
    def translate():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "")).strip()
        if not text:
            return jsonify(error="日本語を入力してください"), 400
        if len(text) > 1000:
            return jsonify(error="入力は1000文字以内にしてください"), 400
        try:
            with model_lock:
                result = task95.translate_texts(
                    model,
                    [text],
                    source_sp,
                    target_sp,
                    device,
                    max_length,
                    batch_size=1,
                    beam_size=beam_size,
                )[0]
            return jsonify(translation=result, beam_size=beam_size)
        except Exception as error:
            app.logger.exception("Translation failed")
            return jsonify(error=f"翻訳に失敗しました: {error}"), 500

    print(f"Model: {checkpoint}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="99: Flask translation server")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--smoke-text",
        help="Translate once through Flask's test client, then exit",
    )
    args = parser.parse_args()
    app = create_app(args.checkpoint, args.beam_size)
    if args.smoke_text:
        response = app.test_client().post("/translate", json={"text": args.smoke_text})
        print(json.dumps(response.get_json(), ensure_ascii=False, indent=2))
        if response.status_code != 200:
            raise SystemExit(1)
        return
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
