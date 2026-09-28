# dataset.md — LiveCodeBench Veri Seti

## 1. LiveCodeBench Nedir?

LiveCodeBench (LCB), LeetCode, Codeforces ve AtCoder'dan toplanan,
a contamination-free code generation benchmark.
Each problem contains a problem statement, input/output examples, and automated test cases. Three difficulty levels: Easy / Medium / Hard.

LCB v6 problems use standard stdin/stdout I/O —
all problems follow the same execution model.

**HuggingFace dataset:** `livecodebench/code_generation_lite`

---

## 2. Dataset Structure

Each example contains the following fields:

```python
{
    "question_id":      "...",
    "question_title":   "Two Sum",
    "question_content": "...",          # HTML — temizlenmeli
    "difficulty":       "easy",         # easy | medium | hard
    "starter_code":     "",             # Usually empty (stdin/stdout problems)
    "public_tests": {                   # 2–5 visible examples
        "input":  ["6\n1 2 3 4 5 6"],
        "output": ["12"]
    },
    "private_tests": {                  # 5–20 gizli test
        "input":  [...],
        "output": [...]
    },
    "platform": "codeforces"
}
```

---

## 3. Test Case Layers

```
public_tests   → 2–5 örnek, problem metninde de gösterilen
private_tests  → 5–20 gizli test, gerçek pass@1 skoru bunlarla
edge_cases     → pipeline tarafından üretilir (public_tests az olunca)

Node 2 Faz 1 (lcb evaluation) : public + private + edge
Node 2 Faz 2 (runtime metrics): sadece public (2–3 test yeterli)
```

---

## 4. Kurulum

```bash
pip install datasets

# For lcb_runner evaluation layer
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench
pip install -e .
```

---

## 5. dataset/loader.py

```python
"""
LiveCodeBench yükleme, temizleme ve test birleştirme modülü.
"""
import re
from datasets import load_dataset


def load_lcb(
    split: str = "test",
    difficulty: str = "all",
    sample_size: int | None = None,
) -> list[dict]:
    """
    LCB veri setini yükler, temizler ve test case'leri birleştirir.

    Args:
        split:       "test" | "val"
        difficulty:  "all" | "easy" | "medium" | "hard"
        sample_size: None = tümü, int = ilk N örnek

    Returns:
        Her biri şu anahtarları içeren dict listesi:
        id, title, content, difficulty, starter_code,
        public_tests, private_tests, all_tests
    """
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split=split,
        trust_remote_code=True,
    )

    rows = []
    for item in ds:
        if difficulty != "all" and item["difficulty"] != difficulty:
            continue

        public  = item.get("public_tests",  {})
        private = item.get("private_tests", {})

        # Merge public + private
        all_inputs  = list(public.get("input",  [])) + \
                      list(private.get("input",  []))
        all_outputs = list(public.get("output", [])) + \
                      list(private.get("output", []))

        rows.append({
            "id":           item["question_id"],
            "title":        item["question_title"],
            "content":      _clean_html(item["question_content"]),
            "difficulty":   item["difficulty"],
            "starter_code": item.get("starter_code", ""),
            "public_tests": {
                "input":  list(public.get("input",  [])),
                "output": list(public.get("output", [])),
            },
            "private_tests": {
                "input":  list(private.get("input",  [])),
                "output": list(private.get("output", [])),
            },
            "all_tests": {
                "input":  all_inputs,
                "output": all_outputs,
            },
        })

    if sample_size is not None:
        rows = rows[:sample_size]

    return rows


def _clean_html(text: str) -> str:
    """HTML tag'lerini temizler, metni düz hale getirir."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;",  " ",  text)
    text = re.sub(r"&lt;",    "<",  text)
    text = re.sub(r"&gt;",    ">",  text)
    text = re.sub(r"&amp;",   "&",  text)
    text = re.sub(r"\n{3,}",  "\n\n", text)
    return text.strip()
```

---

## 6. dataset/splitter.py

```python
"""
Format dönüşümü, istatistik ve problem formatlama.
"""
from collections import Counter
from dataset.loader import load_lcb


def format_problem(row: dict) -> str:
    """
    LCB sorusunu LLM codegen system prompt'una eklenecek formata dönüştürür.

    Sadece public_tests örnekleri gösterilir — private testler gizli kalır.
    """
    parts = [
        f"## Problem: {row['title']}",
        f"Difficulty: {row['difficulty']}",
        "",
        row["content"],
    ]

    # Starter code nadiren dolu olur; varsa ekle
    if row.get("starter_code", "").strip():
        parts += ["", "## Starter Code",
                  "```python", row["starter_code"], "```"]

    # Only public examples are shown
    pub_in  = row["public_tests"].get("input",  [])
    pub_out = row["public_tests"].get("output", [])

    if pub_in:
        parts += ["", "## Examples"]
        for i, (inp, out) in enumerate(zip(pub_in, pub_out), 1):
            parts += [f"**Example {i}:**",
                      f"Input:\n{inp}",
                      f"Output:\n{out}", ""]

    parts += [
        "## Output Format",
        "Read input from stdin. Write answer to stdout.",
        "Output only Python code inside ```python ... ``` markers.",
    ]

    return "\n".join(parts)


def dataset_stats(rows: list[dict]) -> dict:
    counts = Counter(r["difficulty"] for r in rows)
    return {
        "total":  len(rows),
        "easy":   counts.get("easy",   0),
        "medium": counts.get("medium", 0),
        "hard":   counts.get("hard",   0),
    }
```

---

## 7. dataset/edge_cases.py

```python
"""
LCB test sayısı az olan sorular için kural tabanlı edge case üretimi.

Kural: all_tests toplamı < EDGE_CASE_THRESHOLD ise edge case üret.
LLM gerektirmez — problem constraint'lerine göre deterministik üretim.
"""
import ast
import re

EDGE_CASE_THRESHOLD = 8   # Add edge cases when below this count


def needs_edge_cases(row: dict) -> bool:
    """all_tests toplamı eşiğin altındaysa True."""
    return len(row["all_tests"]["input"]) < EDGE_CASE_THRESHOLD


def generate_edge_cases(row: dict) -> list[dict]:
    """
    Problem içeriğinden kural tabanlı edge case'ler üretir.

    Returns:
        [{"input": str, "output": None}, ...]
        output=None → doğru cevap bilinmiyor, sadece crash etmesin diye test edilir

    Not:
        output=None olan case'ler pass/fail için değil,
        runtime error tespiti için kullanılır.
        Node 2'de bu case'ler "crash test" olarak ayrı sayılır.
    """
    cases = []
    content = row["content"].lower()

    # 1. Empty input — applicable to almost every problem
    cases.append({"input": "0\n", "output": None})

    # 2. Single-element list
    if "list" in content or "array" in content or "nums" in content:
        cases.append({"input": "1\n0\n",   "output": None})
        cases.append({"input": "1\n-1\n",  "output": None})

    # 3. Large N — TLE test
    if re.search(r"10\^[456]|1000000|100000", content):
        n = 100_000
        cases.append({
            "input":  f"{n}\n" + " ".join(["1"] * n) + "\n",
            "output": None
        })

    # 4. Negative numbers
    if re.search(r"negative|integer|int", content):
        cases.append({"input": "3\n-1 -2 -3\n", "output": None})

    # 5. All zeros
    cases.append({"input": "3\n0 0 0\n", "output": None})

    return cases


def augment_row(row: dict) -> dict:
    """
    Gerekiyorsa edge case'leri all_tests'e ekler.
    public_tests ve private_tests'e dokunmaz.

    Returns:
        Güncellenmiş row (edge_cases alanı eklenir)
    """
    if not needs_edge_cases(row):
        return {**row, "edge_cases": []}

    edge = generate_edge_cases(row)
    return {**row, "edge_cases": edge}
```

---

## 8. dataset/verify.py

```bash
python dataset/verify.py
```

Also checks that `lcb_runner` is importable — `sandbox/lcb_eval.py` silently falls
back to a crude exact-string-match evaluator (with fabricated `crash_count`/
`test_diversity_score` placeholders) when it isn't installed, which would
otherwise invalidate the thesis's "official LCB evaluator" correctness claim
without anyone noticing.

```python
"""
Veri seti kurulum doğrulama scripti.
"""
import sys
from dataset.loader import load_lcb
from dataset.splitter import dataset_stats, format_problem
from dataset.edge_cases import augment_row, needs_edge_cases


def main():
    print("LiveCodeBench doğrulanıyor...")

    try:
        from lcb_runner.evaluation.utils_execute import check_correctness  # noqa: F401
        print("  lcb_runner: OK (resmi evaluator kullanılabilir)")
    except ImportError:
        print(
            "  UYARI: lcb_runner kurulu değil — sandbox/lcb_eval.py sessizce "
            "fallback evaluator'a düşecek (crash_count/test_diversity_score "
            "gerçek ölçüm değil, sabit değer olacak). Çalıştır: pip install -e LiveCodeBench/"
        )

    try:
        rows = load_lcb(split="test", difficulty="all", sample_size=10)
        print(f"  Yükleme OK — {len(rows)} örnek")
    except Exception as e:
        print(f"  HATA: {e}")
        sys.exit(1)

    stats = dataset_stats(rows)
    print(f"  Dağılım: {stats}")

    # Test case counts
    for row in rows[:3]:
        pub  = len(row["public_tests"]["input"])
        priv = len(row["private_tests"]["input"])
        all_ = len(row["all_tests"]["input"])
        need = needs_edge_cases(row)
        print(f"  {row['id'][:12]:12s} | public={pub:2d} private={priv:2d} "
              f"all={all_:2d} | edge_needed={need}")

    # Format conversion
    sample    = rows[0]
    formatted = format_problem(sample)
    print(f"\n--- Format örneği ({sample['title']}) ---")
    print(formatted[:500])
    print("...")

    # Edge case generation
    aug = augment_row(rows[0])
    print(f"\n  Edge case sayısı (ilk soru): {len(aug['edge_cases'])}")

    print("\nDoğrulama tamamlandı.")


if __name__ == "__main__":
    main()
```

**Expected output:**

```
LiveCodeBench doğrulanıyor...
  Yükleme OK — 10 örnek
  Dağılım: {'total': 10, 'easy': 3, 'medium': 5, 'hard': 2}
  q_001_two_sum  | public= 3 private=12 all=15 | edge_needed=False
  q_002_max_sub  | public= 2 private= 5 all= 7 | edge_needed=True
  q_003_palindro | public= 2 private= 4 all= 6 | edge_needed=True

--- Format örneği (Two Sum) ---
## Problem: Two Sum
Difficulty: easy
...
  Edge case sayısı (ilk soru): 4

Doğrulama tamamlandı.
```

---

## 9. Important Notes

- `private_tests` may be empty for some problems — `all_tests` handles this automatically,
  falling back to public tests only.
- `starter_code` is almost always empty; stdin/stdout problems do not require a template.
- Edge case'lerdeki `output=None` case'ler Node 2'de crash testi olarak
  are counted separately — not included in pass_rate, used only for runtime error detection.
- The `EDGE_CASE_THRESHOLD = 8` value can be tuned during the pilot run.
