from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.checkpoint import load_checkpoint
from nanoqwen.generation import generate
from nanoqwen.tokenizer import ByteTokenizer, apply_chat_template, load_hf_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a nanoqwen chat UI.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hf-tokenizer", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_prompt(
    messages: list[dict[str, str]],
    tokenizer: Any | None,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, Any]:
    if tokenizer is None:
        byte_tokenizer = ByteTokenizer(eos_token_id=eos_token_id or 256)
        text = ""
        for message in messages:
            text += f"{message['role']}: {message['content']}\n"
        text += "assistant:"
        input_ids = torch.tensor([byte_tokenizer.encode(text)], dtype=torch.long)
        return input_ids, byte_tokenizer

    text = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").input_ids, tokenizer


def decode_new_tokens(tokenizer: Any, ids: list[int]) -> str:
    if isinstance(tokenizer, ByteTokenizer):
        return tokenizer.decode(ids)
    return tokenizer.decode(ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    model, _ = load_checkpoint(args.checkpoint, map_location=args.device)
    model.to(args.device).eval()
    tokenizer = load_hf_tokenizer(args.hf_tokenizer) if args.hf_tokenizer else None
    ui_path = Path(__file__).resolve().parents[1] / "nanoqwen" / "ui.html"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self.send_json(HTTPStatus.OK, {"ok": True})
                return
            if self.path not in {"/", "/index.html"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = ui_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = payload["messages"]
                max_new_tokens = int(payload.get("max_new_tokens", 160))
                temperature = float(payload.get("temperature", 0.8))
                top_k = int(payload.get("top_k", 50))

                input_ids, active_tokenizer = make_prompt(
                    messages,
                    tokenizer,
                    model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else None,
                )
                input_ids = input_ids.to(args.device)
                output_ids = generate(
                    model,
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                    do_sample=temperature > 0,
                )
                content = decode_new_tokens(
                    active_tokenizer,
                    output_ids[0, input_ids.shape[1] :].tolist(),
                ).strip()
                self.send_json(HTTPStatus.OK, {"content": content})
            except Exception as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving nanoqwen chat at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
