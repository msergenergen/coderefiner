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

# Optional: lcb_runner (not used for reported results; LCB_EVALUATOR=local)
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
python -m dataset.verify
```

Reports whether `lcb_runner` is installed (informational only — every reported result uses the
pinned local evaluator, `LCB_EVALUATOR=local`), checks that the main-experiment window
2025-01..2025-04 loads 182 problems, and prints test counts for a 10-problem sample.

```python
"""
Dataset installation verification script.
"""
import sys
from dataset.loader import load_lcb
from dataset.splitter import dataset_stats, format_problem
from dataset.edge_cases import augment_row, needs_edge_cases


def main():
    print("Verifying LiveCodeBench...")

    # All reported results use the local LCB-compatible evaluator
    # (LCB_EVALUATOR=local, sandbox/lcb_eval.py::_fallback_evaluate), pinned so
    # every machine grades identically; lcb_runner is optional.
    try:
        from lcb_runner.evaluation.utils_execute import check_correctness  # noqa: F401
        print("  lcb_runner: installed (unused unless LCB_EVALUATOR=lcb_runner)")
    except ImportError:
        print("  lcb_runner: not installed — fine, the local evaluator is used (LCB_EVALUATOR=local)")

    # Main experiment window: LCB 2025-01..2025-04 (docs/dataset.md §10).
    try:
        window = load_lcb(split="test", difficulty="all", date_from="2025-01", date_to="2025-04")
        print(f"  Window 2025-01..2025-04: {len(window)} problems (expected 182)")
    except Exception as e:
        print(f"  ERROR loading window: {e}")
        sys.exit(1)

    try:
        rows = load_lcb(split="test", difficulty="all", sample_size=10)
        print(f"  Load OK — {len(rows)} examples")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    stats = dataset_stats(rows)
    print(f"  Distribution: {stats}")

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
    print(f"\n--- Format example ({sample['title']}) ---")
    print(formatted[:500])
    print("...")

    # Edge case generation
    aug = augment_row(rows[0])
    print(f"\n  Edge cases for first question: {len(aug['edge_cases'])}")

    print("\nVerification complete.")


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

---

## 10. Veri sızıntısı ve zaman penceresi (2026-10-02)

**Durum.** LCB'nin son sürümü `release_v6` (= `release_latest`): Mayıs 2023 – Nisan 2025 arası
1055 problem. Hugging Face veri setinin son güncellemesi 2025-06-05, GitHub deposundaki son
commit 2025-07-16; v6'dan sonra yeni sürüm yayımlanmadı ve bunun nedenine dair resmî bir
açıklama bulamadık. 2026'daki "LCB" tablolarının çoğu hâlâ v6 üzerinden, üretici beyanı puanlar.

**Neden önemli.** LCB'nin sızıntıya karşı temel önlemi, bir modeli yalnızca eğitim verisinin
kesim tarihinden *sonra* yayımlanan problemlerle sınamaktır (Jain vd., arXiv:2403.07974,
Şekil 1). Yayın tarihi, kesim tarihinin üst sınırıdır. Güncel modellerde bu artık uygulanamaz:

| Model | HF yayın tarihi | LCB'nin son problemine göre |
|---|---|---|
| Qwen3.5-9B | 2026-02-27 | Tüm v6 problemleri öncesinde → ezber dışlanamaz |
| gpt-oss-120b | 2025-08-04 | Tüm v6 problemleri öncesinde → ezber dışlanamaz |
| DeepSeek-V3 (ana deney üreticisi) | 2024-12-26 | Ocak–Nisan 2025 problemlerinden önce yayımlanmış → temiz |
| Qwen2.5-72B-Instruct (ana deney yargıcı) | 2024-09-19 | Ocak–Nisan 2025 problemlerinden önce yayımlanmış → temiz |
| Qwen2.5-Coder-32B-Instruct (yalnızca ilk pilot) | 2024-11-06 | temiz |
| Llama-3.3-70B-Instruct (yalnızca ilk pilot) | 2024-11-26 | temiz |

Seed 42 ile çekilen mevcut 150'lik örneğin yıl dağılımı: 2023: 52, 2024: 77, 2025: 21.
v6'da 2025-01 ve sonrası: 182 problem (kolay 45, orta 55, zor 82).

**Etkisi.** Eşleştirilmiş V0–V3 karşılaştırmasında ezber iki koşulu birlikte şişirir; ancak
ilk denemede (ezberden) çözülen problemler arıtıma yer bırakmaz, yani kazanç olduğundan
*küçük* görünebilir. Yargıç da çözümü ezberlemiş olabilir ve yönergeye sızdırabilir.

**Uygulama.** `load_lcb(..., date_from="2025-01", date_to="2025-04")` (ayar: `LCB_DATE_FROM`,
`LCB_DATE_TO`) pencereyi ay bazında, sınırlar dahil uygular; filtre seed'li örneklemeden önce
çalışır ve her satıra `contest_date` (YYYY-MM-DD) eklenir. Ana deneyde `LCB_SAMPLE_SIZE=150`:
pencereden seed 42 ile 150 problem (kolay 40, orta 38, zor 72; Ocak 40, Şubat 42, Mart 56, Nisan
12; 55'i LeetCode). Pencere: 182 problem, 70'i LeetCode
(başlangıç kodlu), hepsinde açık test var.

**Karar.** Ana deney sızıntısız tasarımla yapılır (bkz. `docs/ablation.md` §7): üretici ve
yargıç Ocak 2025'ten önce yayımlanmış modeller, problemler yalnızca 2025-01 – 2025-04
penceresinden. Qwen3.5-9B ile yerelde alınan sonuçlar "güncel akıl yürütme modeli" ek bulgusu
olarak kalır ve ezber riski hem bildiride hem tezde sınırlılık olarak açıkça yazılır.
