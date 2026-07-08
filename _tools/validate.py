#!/usr/bin/env python3
"""验证 minimind-tutorial 所有章节的完整性。"""
import json, os, sys, glob

BASE = "/home/minimind-tutorial"
ERRORS = []
WARNINGS = []
PASS = 0

def ok(msg=""):
    global PASS
    PASS += 1

def warn(msg):
    WARNINGS.append(msg)

def err(msg):
    ERRORS.append(msg)

for ch_num in range(1, 16):
    ch = f"ch{ch_num:02d}"
    chdir = os.path.join(BASE, ch, "01_main-chapter-code")
    if not os.path.isdir(chdir):
        err(f"{ch}: directory missing {chdir}")
        continue

    main_nb = "ch{:02d}.ipynb".format(ch_num)
    expected_files = [main_nb, "exercise-solutions.ipynb", "{}.md".format(ch), "README.md"]
    if ch_num == 1:
        expected_files.append("big-picture.ipynb")
    else:
        try:
            summary_candidates = [f for f in os.listdir(chdir) if f.endswith(".ipynb") and f != main_nb and f != "exercise-solutions.ipynb"]
            if summary_candidates:
                expected_files.append(summary_candidates[0])
        except OSError:
            pass

    for ef in expected_files:
        path = os.path.join(chdir, ef)
        if not os.path.exists(path):
            err(f"{ch}: missing {ef}")
            continue

    for nb_file in glob.glob(os.path.join(chdir, "*.ipynb")):
        nb_name = os.path.basename(nb_file)
        try:
            with open(nb_file, "r", encoding="utf-8") as f:
                nb = json.load(f)
            cells = nb.get("cells", [])
            n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
            n_code = sum(1 for c in cells if c["cell_type"] == "code")
            ratio = n_md / max(n_code, 1) if n_code else 0

            if nb_name == main_nb:
                sources = []
                for c in cells:
                    sources.extend(c["source"].split("\n") if isinstance(c["source"], str) else c["source"])
                full_text = "\n".join(sources)
                if not full_text.startswith(f"# 第"):
                    warn(f"{ch}/ch{ch_num}.ipynb: doesn't start with '# 第'")
                if "Summary and takeaways" not in full_text:
                    warn(f"{ch}/ch{ch_num}.ipynb: missing 'Summary and takeaways'")
                if n_code > 0 and ratio < 2.0:
                    warn(f"{ch}/ch{ch_num}.ipynb: md:code ratio {ratio:.1f}:1 (target ≥ 2:1)")
                ok()
            elif nb_name == "exercise-solutions.ipynb":
                sources = []
                for c in cells:
                    sources.extend(c["source"].split("\n") if isinstance(c["source"], str) else c["source"])
                full_text = "\n".join(sources)
                n_exercises = full_text.count("## Exercise")
                if n_exercises < 3:
                    warn(f"{ch}/exercise-solutions.ipynb: only {n_exercises} exercises (target 3)")
                ok()
            else:
                ok()
        except json.JSONDecodeError as e:
            err(f"{ch}/{nb_name}: INVALID JSON: {e}")

    md_path = os.path.join(chdir, f"{ch}.md")
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        if not any(kw in md_content for kw in ["model_minimind", "train_", "lm_dataset", "serve_openai", "convert_model", "eval_llm", "rollout_engine", "model_lora"]) and ch_num > 1:
            warn(f"{ch}/{ch}.md: no minimind file references found")
        ok()

print("=" * 60)
print(f"PASSED: {PASS}  WARNINGS: {len(WARNINGS)}  ERRORS: {len(ERRORS)}")
print("=" * 60)
if WARNINGS:
    print("\n⚠️  WARNINGS:")
    for w in WARNINGS:
        print(f"  {w}")
if ERRORS:
    print("\n❌ ERRORS:")
    for e in ERRORS:
        print(f"  {e}")
    sys.exit(1)
else:
    print("\n✅ All chapters valid.")
