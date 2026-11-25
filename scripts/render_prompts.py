#!/usr/bin/env python3
"""Render prompts/suite.jsonl into prompts/rendered/<category>__<name>.txt.

Chat prompts go through the model's own Jinja chat template via a running llama-server's /apply-template endpoint,
so the special tokens are exactly what the server would send. Raw prompts are written unchanged.

usage: scripts/render_prompts.py [--server http://127.0.0.1:18089] [--suite prompts/suite.jsonl] [--out prompts/rendered]
"""
import argparse
import json
import pathlib
import urllib.request


def apply_template(server: str, text: str, effort: str) -> str:
    body = {"messages": [{"role": "user", "content": text}], "chat_template_kwargs": {"reasoning_effort": effort}}
    req = urllib.request.Request(server + "/apply-template", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["prompt"]


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:18089")
    ap.add_argument("--suite", default=str(root / "prompts" / "suite.jsonl"))
    ap.add_argument("--out", default=str(root / "prompts" / "rendered"))
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.txt"):
        old.unlink()
    n = 0
    for line in pathlib.Path(a.suite).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        text = apply_template(a.server, p["text"], p.get("effort", "none")) if p["mode"] == "chat" else p["text"]
        (out / f"{p['category']}__{p['name']}.txt").write_text(text, encoding="utf-8")
        n += 1
    print(f"rendered {n} prompts into {out}")


if __name__ == "__main__":
    main()
