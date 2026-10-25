"""006 Help assistant: a small model on this computer (Ollama) answers from config/help.yaml text only.

It never sees app data (clips, labels, AI suggestions, reviews, reviewer names, statuses, results, paths, video names):
only the matched help entries, the question and the name of the tab it was asked from (used to prefer that tab's
entries). Nothing is logged. The only network call is to Ollama on 127.0.0.1:11434.
"""
import json
import math
import re
import urllib.error
import urllib.request
from urllib.parse import urlsplit

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:1.5b"
MAX_QUESTION = 300
TIMEOUT_S = 20
UNAVAILABLE = ("The help assistant isn't available on this computer (start Ollama to use it). "
               "You can still browse the Help tab.")
NO_MODEL = ("The help assistant isn't available on this computer (start Ollama and make sure the model {model} is "
            "installed). You can still browse the Help tab.")
DONT_KNOW = "I don't know that one — try the questions below."
SYSTEM = ("You answer questions from TPA staff about this app. Answer ONLY from the help entries below, in 1-3 plain "
          "sentences. Never invent numbers, names or features. If the entries do not answer the question, reply "
          f'exactly: "{DONT_KNOW}"')
TAB_NAMES = {"scorecard": "Scorecard", "search": "Search moments", "review": "Review queue",
             "process": "Process a video", "blind": "Blind labels", "export": "Export", "help": "Help"}
TAB_BOOST = 2.0  # entries tagged for the tab the question was asked from
_STOP = set("a an and are can do does for how i if in is it me my of on or the this to what when where which who "
            "why with you your".split())
# no proxies: an HTTP(S)_PROXY in the environment must never route the request off this computer
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _words(text: str) -> list[str]:
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


def top_entries(items: list[dict], question: str, k: int = 3, tab: str | None = None) -> list[dict]:
    """The k entries with the highest TF-IDF word overlap with the question (question text counts double). Entries
    tagged for `tab` score TAB_BOOST times higher; an entry still needs some overlap to be picked."""
    docs = [_words(f"{it['question']} {it['question']} {it['answer']}") for it in items]
    df = {w: sum(w in d for d in docs) for w in set(_words(question))}
    idf = {w: math.log(len(docs) / n) for w, n in df.items() if n}
    boost = [TAB_BOOST if tab and tab in (it.get("tabs") or []) else 1.0 for it in items]
    scored = [(sum(d.count(w) * v for w, v in idf.items()) * b, b, i) for i, (d, b) in enumerate(zip(docs, boost))]
    # on a tie, the entry tagged for this tab comes first, then file order
    return [items[i] for s, _, i in sorted(scored, key=lambda t: (-t[0], -t[1], t[2]))[:k] if s > 0]


def messages(entries: list[dict], question: str, tab: str | None = None) -> list[dict]:
    """The whole prompt: the system rule, the tab's name, the matched help.yaml entries and the question, nothing
    else."""
    text = "\n\n".join(f"Q: {it['question']}\nA: {it['answer']}" for it in entries)
    where = f"\n\nCurrent tab: {TAB_NAMES[tab]}" if tab in TAB_NAMES else ""
    return [{"role": "system", "content": f"{SYSTEM}{where}\n\nHelp entries:\n{text}"},
            {"role": "user", "content": question[:MAX_QUESTION]}]


def _post(url: str, payload: dict) -> dict:
    parts = urlsplit(url)
    if (parts.scheme, parts.hostname, parts.port) != ("http", "127.0.0.1", 11434):
        raise ValueError("the help assistant only talks to Ollama on 127.0.0.1:11434")
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read())


def ask(question: str, items: list[dict], model: str = DEFAULT_MODEL, tab: str | None = None) -> tuple[str, list[str]]:
    """(answer, matched questions). Unavailable message if Ollama is not running or the model is missing. `tab` is
    only the tab's name: no other app state is ever passed in."""
    question = (question or "").strip()[:MAX_QUESTION]
    if not question:
        return "", []
    tab = tab if tab in TAB_NAMES else None
    entries = top_entries(items, question, tab=tab)
    if not entries:
        return DONT_KNOW, []
    payload = {"model": model, "messages": messages(entries, question, tab), "stream": False,
               "options": {"temperature": 0.2, "num_predict": 120}}
    try:
        answer = _post(OLLAMA_URL, payload)["message"]["content"].strip()
    except urllib.error.HTTPError as e:  # Ollama is running; 404 means the model is not installed
        return (NO_MODEL.format(model=model) if e.code == 404 else UNAVAILABLE), []
    except (OSError, ValueError, KeyError, TypeError):  # refused, timeout, HTTP 404 (no model), bad reply
        return UNAVAILABLE, []
    return answer or DONT_KNOW, [it["question"] for it in entries]
