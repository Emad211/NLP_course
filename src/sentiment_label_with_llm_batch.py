from __future__ import annotations

import os
import json
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import pandas as pd
from openai import OpenAI


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
IN_JSONL = BASE_DIR / "processed_data" / "sentiment" / "labeling_pool.jsonl"
IN_POOL_CSV = BASE_DIR / "processed_data" / "sentiment" / "labeling_pool.csv"

OUT_DIR = BASE_DIR / "processed_data" / "sentiment"
OUT_LABELS_JSONL = OUT_DIR / "llm_labels.jsonl"
OUT_LABELED_CSV = OUT_DIR / "labeling_pool_llm_labeled.csv"
OUT_DEBUG_LOG = OUT_DIR / "llm_label_debug.log"


# =========================
# Config
# =========================
@dataclass
class Config:
    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    base_url: str = os.environ.get("OPENAI_BASE_URL", "https://api.gapgpt.app/v1")
    model: str = os.environ.get("LLM_MODEL", "gemini-3-flash-preview")

    # batching
    batch_size: int = int(os.environ.get("LLM_BATCH_SIZE", "30"))
    max_chars_per_batch: int = int(os.environ.get("LLM_MAX_CHARS_PER_BATCH", "12000"))
    max_text_chars_per_item: int = int(os.environ.get("LLM_MAX_TEXT_CHARS_PER_ITEM", "700"))

    # generation controls
    temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0"))
    max_output_tokens: int = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "1800"))

    # robustness
    max_retries: int = int(os.environ.get("LLM_MAX_RETRIES", "3"))
    backoff_base: float = float(os.environ.get("LLM_BACKOFF_BASE", "1.7"))

    # if we extracted >= this ratio of items from a truncated output, accept and retry only missing
    min_partial_ok_ratio: float = float(os.environ.get("LLM_MIN_PARTIAL_OK_RATIO", "0.25"))

    # allow bisect down to this size; then go 1-by-1 only if still failing
    min_bisect_batch: int = int(os.environ.get("LLM_MIN_BISECT_BATCH", "2"))

    # debug
    save_raw_on_error: bool = os.environ.get("LLM_SAVE_RAW_ON_ERROR", "1") == "1"
    raw_snippet_len: int = int(os.environ.get("LLM_RAW_SNIPPET_LEN", "2000"))

    # resume
    skip_already_ok: bool = os.environ.get("LLM_SKIP_ALREADY_OK", "1") == "1"

    # optional limit for testing
    max_items: int = int(os.environ.get("LLM_MAX_ITEMS", "0"))  # 0 => all


CFG = Config()


# =========================
# Logging
# =========================
def now_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{now_ts()}] {msg.rstrip()}\n")


# =========================
# Input cleanup
# =========================
SPACE_RE = re.compile(r"\s+")
def compact_text(t: str) -> str:
    t = (t or "").replace("\u0000", " ")
    t = t.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    t = SPACE_RE.sub(" ", t).strip()
    return t


# =========================
# Output cleanup / JSON salvage
# =========================
FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)
BAD_FLOAT_RE = re.compile(r'(:\s*)(-?\d+)\.(\s*[,}\]])')   # 0.  -> 0.0
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

def sanitize_output(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = FENCE_RE.sub("", s).strip()
    s = BAD_FLOAT_RE.sub(r"\g<1>\g<2>.0\g<3>", s)
    s = TRAILING_COMMA_RE.sub(r"\1", s)
    return s.strip()


def try_parse_json_array(s: str) -> Optional[list]:
    if not s:
        return None
    s = sanitize_output(s)

    # direct
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else None
    except Exception:
        pass

    # substring [ ... ]
    i = s.find("[")
    j = s.rfind("]")
    if i >= 0 and j > i:
        sub = s[i:j+1]
        try:
            obj = json.loads(sub)
            return obj if isinstance(obj, list) else None
        except Exception:
            return None

    return None


def extract_complete_objects_from_text(s: str) -> List[dict]:
    """
    If output is truncated and not a valid JSON array, salvage any COMPLETE {...} objects.
    Works by scanning braces with string/escape awareness.
    """
    s = sanitize_output(s)
    if not s:
        return []

    # focus on content after first '[' if exists
    i = s.find("[")
    if i >= 0:
        s = s[i+1:]

    objs = []
    in_str = False
    esc = False
    depth = 0
    start = None

    for idx, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        chunk = s[start:idx+1]
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, dict):
                                objs.append(obj)
                        except Exception:
                            pass
                        start = None

    return objs


# =========================
# Resume: consider done only if status=ok
# =========================
def is_ok_record(obj: Dict[str, Any]) -> bool:
    try:
        if obj.get("status") != "ok":
            return False
        iid = str(obj.get("item_id", "")).strip()
        if not iid:
            return False
        lab = int(obj.get("label_llm"))
        conf = float(obj.get("llm_confidence"))
        if lab not in (-1, 0, 1):
            return False
        if not (0.0 <= conf <= 1.0):
            return False
        return True
    except Exception:
        return False


def load_done_ok(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if is_ok_record(obj):
                    done.add(str(obj["item_id"]))
            except Exception:
                continue
    return done


# =========================
# Prompt
# =========================
def build_prompt(items: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    Keep instructions SHORT to reduce token budget.
    Use confidence_int to avoid JSON float issues.
    """
    instructions = (
        "Return ONLY a valid JSON array. No markdown, no explanation.\n"
        "Each element MUST be: {\"item_id\":\"...\",\"label\":-1|0|1,\"confidence_int\":0..100}\n"
        "Labels: 1=positive, 0=neutral/mixed/info, -1=negative.\n"
        "Service complaints (waiting, secretary, scheduling, phone, cost) => negative.\n"
        "Mixed praise+complaint: if unclear => 0.\n"
        "Text-only info like 'ام ار ای نوشتن' => 0.\n"
        "Output MUST start with '[' and end with ']'. confidence_int must be INTEGER.\n"
    )
    payload = [{"item_id": it["item_id"], "text": it["text"]} for it in items]
    input_text = json.dumps(payload, ensure_ascii=False)
    return instructions, input_text


# =========================
# API call (Responses then Chat)
# =========================
def call_model(client: OpenAI, instructions: str, input_text: str) -> str:
    # Try Responses
    try:
        resp = client.responses.create(
            model=CFG.model,
            instructions=instructions,
            input=input_text,
            temperature=CFG.temperature,
            max_output_tokens=CFG.max_output_tokens,
        )
        out = getattr(resp, "output_text", None)
        if out:
            return str(out)
    except Exception as e:
        log(f"responses.create failed: {type(e).__name__}: {str(e)[:240]}")

    # Fallback Chat Completions
    comp = client.chat.completions.create(
        model=CFG.model,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": input_text},
        ],
        temperature=CFG.temperature,
        max_tokens=CFG.max_output_tokens,
    )
    return (comp.choices[0].message.content or "")


# =========================
# Mapping/validation
# =========================
def map_objects_to_results(
    items: List[Dict[str, str]],
    objs: List[dict],
    batch_id: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    wanted = [it["item_id"] for it in items]
    wanted_set = set(wanted)

    got: Dict[str, dict] = {}
    for o in objs:
        if not isinstance(o, dict):
            continue
        iid = str(o.get("item_id", "")).strip()
        if iid in wanted_set and iid not in got:
            got[iid] = o

    results: List[Dict[str, Any]] = []
    missing: List[str] = []

    for iid in wanted:
        o = got.get(iid)
        if o is None:
            missing.append(iid)
            continue

        # label
        try:
            lab = int(o.get("label"))
        except Exception:
            missing.append(iid)
            continue
        if lab not in (-1, 0, 1):
            missing.append(iid)
            continue

        # confidence_int -> float 0..1
        ci_raw = o.get("confidence_int", 50)
        try:
            ci = int(str(ci_raw).strip())
        except Exception:
            ci = 50
        ci = max(0, min(100, ci))
        conf = ci / 100.0

        results.append(
            {
                "item_id": iid,
                "label_llm": lab,
                "llm_confidence": conf,
                "confidence_int": ci,
                "model": CFG.model,
                "status": "ok",
                "batch_id": batch_id,
                "ts_utc": now_ts(),
            }
        )

    return results, missing


def backoff_sleep(attempt: int) -> None:
    t = CFG.backoff_base ** max(0, attempt - 1)
    time.sleep(min(20.0, t))


def label_items_recursive(client: OpenAI, items: List[Dict[str, str]], batch_id: str, depth: int = 0) -> List[Dict[str, Any]]:
    """
    Core engine:
    - Calls model for a batch.
    - If full JSON array parses: great.
    - If not: salvage complete objects from truncated output.
      * If we salvaged enough, accept them and retry missing only.
      * Else retry/bisect.
    - Bisect down to size=CFG.min_bisect_batch; if still fails, go 1-by-1 as last resort.
    """
    if not items:
        return []

    # retries on same batch
    last_raw = ""
    last_err: Optional[Exception] = None

    for attempt in range(1, CFG.max_retries + 1):
        try:
            instructions, input_text = build_prompt(items)
            raw = call_model(client, instructions, input_text)
            last_raw = raw

            # 1) try full parse
            arr = try_parse_json_array(raw)
            if isinstance(arr, list):
                results, missing = map_objects_to_results(items, arr, batch_id=f"{batch_id}.d{depth}.full")
                # if missing due to model skipping some, retry only missing (rare)
                if missing:
                    miss_items = [it for it in items if it["item_id"] in set(missing)]
                    res2 = label_items_recursive(client, miss_items, batch_id=batch_id + ".missing", depth=depth + 1)
                    return results + res2
                return results

            # 2) salvage objects
            salvaged = extract_complete_objects_from_text(raw)
            if salvaged:
                results, missing = map_objects_to_results(items, salvaged, batch_id=f"{batch_id}.d{depth}.salv")
                ok_ratio = len(results) / len(items)

                log(f"SALVAGE batch={batch_id} depth={depth} size={len(items)} ok={len(results)} ratio={ok_ratio:.2f}")

                if ok_ratio >= CFG.min_partial_ok_ratio:
                    # accept what we have; retry missing only (with recursion)
                    miss_set = set(missing)
                    miss_items = [it for it in items if it["item_id"] in miss_set]
                    res2 = label_items_recursive(client, miss_items, batch_id=batch_id + ".rem", depth=depth + 1)
                    return results + res2

            raise ValueError("No valid JSON array, and salvage insufficient.")

        except Exception as e:
            last_err = e
            if CFG.save_raw_on_error and last_raw:
                log(f"RAW_SNIP batch={batch_id} depth={depth} attempt={attempt}:\n{sanitize_output(last_raw)[:CFG.raw_snippet_len]}")
            log(f"ERROR batch={batch_id} depth={depth} attempt={attempt}/{CFG.max_retries} size={len(items)}: {type(e).__name__}: {str(e)[:240]}")
            backoff_sleep(attempt)

    # If we get here, retries failed -> bisect or 1-by-1
    n = len(items)
    if n > CFG.min_bisect_batch:
        mid = n // 2
        log(f"BISECT batch={batch_id} depth={depth} size={n} -> {mid}+{n-mid} (reason={type(last_err).__name__})")
        left = label_items_recursive(client, items[:mid], batch_id=batch_id + ".L", depth=depth + 1)
        right = label_items_recursive(client, items[mid:], batch_id=batch_id + ".R", depth=depth + 1)
        return left + right

    # last resort: 1-by-1 (still not "دونه دونه" برای کل دیتاست؛ فقط برای batchهای مشکل‌دار)
    if n == 1:
        it = items[0]
        log(f"FALLBACK_1BY1 item_id={it['item_id']} batch={batch_id} depth={depth}")
        # try one last time with max_retries=1 effectively
        try:
            instructions, input_text = build_prompt(items)
            raw = call_model(client, instructions, input_text)
            arr = try_parse_json_array(raw)
            if isinstance(arr, list):
                results, missing = map_objects_to_results(items, arr, batch_id=f"{batch_id}.one.full")
                if results:
                    return results
            salvaged = extract_complete_objects_from_text(raw)
            if salvaged:
                results, _ = map_objects_to_results(items, salvaged, batch_id=f"{batch_id}.one.salv")
                if results:
                    return results
        except Exception:
            pass

        # ultimate fallback record (NOT ok)
        return [{
            "item_id": it["item_id"],
            "label_llm": 0,
            "llm_confidence": 0.0,
            "confidence_int": 0,
            "model": CFG.model,
            "status": "ultimate_fallback",
            "batch_id": batch_id,
            "ts_utc": now_ts(),
        }]

    # n == 2 and still failing: go one-by-one
    out = []
    out.extend(label_items_recursive(client, [items[0]], batch_id=batch_id + ".s1", depth=depth + 1))
    out.extend(label_items_recursive(client, [items[1]], batch_id=batch_id + ".s2", depth=depth + 1))
    return out


# =========================
# Batching
# =========================
def make_batches(items: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    batches: List[List[Dict[str, str]]] = []
    cur: List[Dict[str, str]] = []
    cur_chars = 0

    for it in items:
        add = len(it["text"]) + len(it["item_id"]) + 10
        if cur and (len(cur) >= CFG.batch_size or (cur_chars + add) > CFG.max_chars_per_batch):
            batches.append(cur)
            cur = []
            cur_chars = 0
        cur.append(it)
        cur_chars += add

    if cur:
        batches.append(cur)

    return batches


# =========================
# Main
# =========================
def main():
    if not CFG.api_key:
        raise RuntimeError("OPENAI_API_KEY env var is not set. Do NOT hardcode keys in source.")
    if not IN_JSONL.exists():
        raise FileNotFoundError(IN_JSONL)
    if not IN_POOL_CSV.exists():
        raise FileNotFoundError(IN_POOL_CSV)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=CFG.api_key, base_url=CFG.base_url)

    done_ok = load_done_ok(OUT_LABELS_JSONL) if (CFG.skip_already_ok and OUT_LABELS_JSONL.exists()) else set()
    log(f"START model={CFG.model} base_url={CFG.base_url} done_ok={len(done_ok)} batch_size={CFG.batch_size} max_chars={CFG.max_chars_per_batch}")

    # Load items
    items: List[Dict[str, str]] = []
    with IN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            item_id = str(obj.get("item_id", "")).strip()
            text = compact_text(str(obj.get("text", "")).strip())
            if not item_id or not text:
                continue
            if item_id in done_ok:
                continue
            if CFG.max_text_chars_per_item > 0 and len(text) > CFG.max_text_chars_per_item:
                text = text[:CFG.max_text_chars_per_item].strip()
            items.append({"item_id": item_id, "text": text})
            if CFG.max_items > 0 and len(items) >= CFG.max_items:
                break

    batches = make_batches(items)
    print(f"to_label_items={len(items):,}  batches={len(batches):,}  already_done_ok={len(done_ok):,}")

    if not items:
        print("Nothing to label. Exiting.")
        return

    # Label and write incremental JSONL
    written = 0
    with OUT_LABELS_JSONL.open("a", encoding="utf-8") as f_out:
        for bi, batch in enumerate(batches, start=1):
            batch_id = f"b{bi:05d}"
            res = label_items_recursive(client, batch, batch_id=batch_id, depth=0)
            for r in res:
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            f_out.flush()

            written += len(res)
            if bi % 5 == 0 or bi == 1:
                print(f"progress batches={bi}/{len(batches)} records_written~={written:,}")

    # Merge into CSV
    pool = pd.read_csv(IN_POOL_CSV, encoding="utf-8")

    labels: List[Dict[str, Any]] = []
    with OUT_LABELS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                labels.append(json.loads(line))
            except Exception:
                continue

    lab = pd.DataFrame(labels)
    if not lab.empty:
        lab = lab.sort_values(["ts_utc"]).drop_duplicates("item_id", keep="last")

    out = pool.merge(lab, on="item_id", how="left")
    out.to_csv(OUT_LABELED_CSV, index=False, encoding="utf-8-sig")

    print(f"saved={OUT_LABELED_CSV.resolve()}")
    print(f"labels_jsonl={OUT_LABELS_JSONL.resolve()}")
    print(f"debug_log={OUT_DEBUG_LOG.resolve()}")


if __name__ == "__main__":
    main()