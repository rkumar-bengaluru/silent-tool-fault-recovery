import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

PRICES = {"claude-haiku-5-5": (0.10, 0.50)}  # USD per 1M tokens (input, output)


class BudgetExceeded(Exception):
    pass


class ModelUnavailable(Exception):
    pass


@dataclass
class Reply:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False


def cache_key(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


class Cache:
    def __init__(self, dir: Path):
        self.dir = Path(dir)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, key: str, value: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))


class CostTracker:
    # shortcut: reservations live in memory (one process per run), spend lives in the file.
    def __init__(self, path: Path, warn_at: float = 40.0, stop_at: float = 50.0):
        self.path, self.warn_at, self.stop_at = Path(path), warn_at, stop_at
        self._reserved: dict[str, float] = {}

    def _entries(self) -> list[dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def spent(self) -> float:
        return sum(e["usd"] for e in self._entries())

    def charge(self, usd: float, run_id: str) -> None:
        entries, before = self._entries(), self.spent()
        entries.append({"run_id": run_id, "usd": usd, "t": time.time()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=1))
        if before < self.warn_at <= before + usd:
            print(f"WARNING: Claude spend ${before + usd:.2f} crossed ${self.warn_at:.2f} (hard stop ${self.stop_at:.2f})")

    def reserve(self, usd: float, run_id: str) -> None:
        self._reserved[run_id] = usd

    def release(self, run_id: str) -> None:
        self._reserved.pop(run_id, None)

    def check(self, est_usd: float, *, reservation_id: str | None = None) -> None:
        """Raise before a call that would pass the hard stop. With reservation_id (a final run), that run's own
        reservation is excluded from the held total, and its own spend so far must not exceed it."""
        held = sum(v for k, v in self._reserved.items() if k != reservation_id)
        total = self.spent() + held + est_usd
        if total > self.stop_at:
            raise BudgetExceeded(f"${total:.2f} would exceed the ${self.stop_at:.2f} hard stop")
        if reservation_id in self._reserved:
            own = sum(e["usd"] for e in self._entries() if e["run_id"] == reservation_id)
            if own > self._reserved[reservation_id]:
                raise BudgetExceeded(f"run spend ${own:.4f} exceeds its ${self._reserved[reservation_id]:.4f} reservation")


def _err_text(resp) -> str:
    try:
        return str(resp.json().get("error", ""))
    except (ValueError, AttributeError):
        return ""


class OllamaChat:
    def __init__(self, model: str, *, base_url: str = "http://localhost:11434", cache: Cache,
                 client: httpx.Client | None = None, sleep=time.sleep):
        self.model, self.base_url, self.cache, self.sleep = model, base_url, cache, sleep
        self.http = client or httpx.Client(timeout=300)

    def chat(self, messages: list[dict], *, seed: int, temperature: float = 0.7) -> Reply:
        body = {"model": self.model, "messages": messages, "stream": False, "format": "json", "think": False,
                "options": {"temperature": temperature, "seed": seed}}
        key = cache_key(body)
        if hit := self.cache.get(key):
            return Reply(**hit, cached=True)
        for delay in (2, 8, 30, None):  # transient 5xx / timeouts: retry with backoff, then give up
            try:
                resp = self.http.post(f"{self.base_url}/api/chat", json=body)
            except httpx.ConnectError as e:
                raise ModelUnavailable(f"Ollama not reachable at {self.base_url}") from e
            except (httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                err = repr(e)
            else:
                if resp.status_code < 500:
                    break
                if "token repeat limit" in _err_text(resp):  # deterministic model degeneration, not unavailability
                    reply = Reply("", 0, 0, 0.0)
                    self.cache.put(key, {k: v for k, v in asdict(reply).items() if k != "cached"})
                    return reply
                err = f"HTTP {resp.status_code}"
            if delay is None:
                raise ModelUnavailable(f"Ollama failed after 3 retries: {err}")
            self.sleep(delay)
        resp.raise_for_status()
        d = resp.json()
        reply = Reply(d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0), 0.0)
        self.cache.put(key, {k: v for k, v in asdict(reply).items() if k != "cached"})
        return reply


class ClaudeChat:
    def __init__(self, model: str, *, cache: Cache, tracker: CostTracker, run_id: str, final: bool = False,
                 client=None):
        PRICES[model]  # unknown model -> KeyError before any spend
        self.model, self.cache, self.tracker, self.run_id, self.final = model, cache, tracker, run_id, final
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.client = client
        self._last_cost = 0.01

    def chat(self, messages: list[dict], *, seed: int, temperature=None) -> Reply:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
        rest = messages[1:] if system is not None else messages
        key = cache_key({"model": self.model, "system": system, "messages": rest, "seed": seed})
        if hit := self.cache.get(key):
            return Reply(**hit, cached=True)
        self.tracker.check(self._last_cost, reservation_id=self.run_id if self.final else None)
        kw = {"system": system} if system is not None else {}
        resp = self.client.messages.create(model=self.model, max_tokens=1024, messages=rest,
                                           output_config={"effort": "low"}, thinking={"type": "disabled"}, **kw)
        p_in, p_out = PRICES[self.model]
        u = resp.usage
        text = "".join(b.text for b in resp.content if b.type == "text")
        reply = Reply(text, u.input_tokens, u.output_tokens, (u.input_tokens * p_in + u.output_tokens * p_out) / 1e6)
        self.tracker.charge(reply.cost_usd, self.run_id)
        self._last_cost = reply.cost_usd
        self.cache.put(key, {k: v for k, v in asdict(reply).items() if k != "cached"})
        return reply
