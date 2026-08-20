"""Bridge from the WSL conda env to the Ollama server on the Windows host.

Ollama binds 127.0.0.1 on Windows, so WSL2 cannot reach it over TCP (same
situation as Postgres). Rather than ask the user to rebind the service, this
uses WSL interop: WSL can execute Windows binaries, so we shell out to Windows
`curl.exe`, which resolves 127.0.0.1 in the Windows network namespace.

Everything therefore stays on the machine -- no patient text is sent to a
hosted API.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

WIN_CURL = "/mnt/c/Windows/System32/curl.exe"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


def _curl() -> str:
    if os.path.exists(WIN_CURL):
        return WIN_CURL
    found = shutil.which("curl.exe") or shutil.which("curl")
    if not found:
        raise RuntimeError(
            "No curl available. Expected Windows curl at %s (WSL interop)." % WIN_CURL)
    return found


class OllamaError(RuntimeError):
    pass


@dataclass
class Ollama:
    model: str = "medgemma:latest"
    temperature: float = 0.0
    num_predict: int = 512
    timeout: int = 300
    retries: int = 2

    def _post(self, path: str, payload: dict) -> dict:
        # The payload goes in on stdin (`-d @-`). A temp file would not work:
        # curl.exe is a Windows binary and cannot open a Linux /tmp path, and
        # inlining the JSON would break on clinical text containing quotes.
        body = json.dumps(payload).encode("utf-8")
        last = None
        for attempt in range(self.retries + 1):
            proc = subprocess.run(
                [_curl(), "-s", "-m", str(self.timeout),
                 "-H", "Content-Type: application/json",
                 f"{OLLAMA_URL}{path}", "-d", "@-"],
                input=body, capture_output=True)
            if proc.returncode != 0:
                last = f"curl exit {proc.returncode}: {proc.stderr[:200]!r}"
            else:
                try:
                    return json.loads(proc.stdout.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    last = f"non-JSON reply: {proc.stdout[:200]!r}"
            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
        raise OllamaError(f"{path} failed after {self.retries + 1} tries: {last}")

    # -- text generation ---------------------------------------------------
    def generate(self, prompt: str, system: str | None = None,
                 fmt: dict | str | None = None) -> str:
        payload = {
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": self.temperature,
                        "num_predict": self.num_predict},
        }
        if system:
            payload["system"] = system
        if fmt is not None:
            payload["format"] = fmt
        return self._post("/api/generate", payload).get("response", "")

    @staticmethod
    def _parse(raw: str) -> dict | None:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            start, end = raw.find("{"), raw.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    def generate_json(self, prompt: str, schema: dict,
                      system: str | None = None) -> dict | None:
        """JSON output, with a fallback path for reasoning models.

        Ollama's structured output (`format`) is incompatible with reasoning
        models such as gpt-oss: they emit into a separate `thinking` field and
        `response` comes back EMPTY, which silently scores as "prescribed
        nothing" rather than raising. So when constrained decoding yields
        nothing, retry unconstrained with a much larger token budget (the model
        must finish reasoning before it emits an answer) and parse the JSON out
        of the free text.
        """
        parsed = self._parse(self.generate(prompt, system=system, fmt=schema))
        if parsed is not None:
            return parsed

        payload = {
            "model": self.model,
            "prompt": prompt + "\n\nReply with ONLY the JSON object.",
            "stream": False,
            "options": {"temperature": self.temperature,
                        "num_predict": max(self.num_predict, 1500)},
        }
        if system:
            payload["system"] = system
        out = self._post("/api/generate", payload)
        return self._parse(out.get("response", ""))

    # -- embeddings --------------------------------------------------------
    def embed(self, texts: list[str], model: str = "nomic-embed-text:latest"):
        out = self._post("/api/embed", {"model": model, "input": texts})
        return out.get("embeddings", [])

    # -- introspection -----------------------------------------------------
    @staticmethod
    def available() -> list[str]:
        try:
            proc = subprocess.run([_curl(), "-s", "-m", "20", f"{OLLAMA_URL}/api/tags"],
                                  capture_output=True, text=True)
            return [m["name"] for m in json.loads(proc.stdout).get("models", [])]
        except Exception:
            return []


if __name__ == "__main__":
    print("models:", Ollama.available())
    o = Ollama()
    print("medgemma:", o.generate("Reply with exactly: OK")[:80])
