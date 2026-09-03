"""
Menu launcher -- the easy way to run this project.

Everything in this artefact is a separate stage with its own prerequisites,
which is correct for a reproducible pipeline and awkward for a person sitting
in front of it. This script removes that friction: it inspects what has
already been produced on disk, prints a menu of what can be run right now,
explains what is blocked and why, and then runs the chosen stage for you.

    python run.py

No arguments, no flags, nothing to remember. In VS Code you can also just
open this file and press the Run button.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

BOLD, DIM, GREEN, YELLOW, RED, BLUE, OFF = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m")


# --------------------------------------------------------------------------- #
# What already exists on disk
# --------------------------------------------------------------------------- #
def state() -> dict:
    ckpts = list((RESULTS / "checkpoints").glob("*.pt")) if (RESULTS / "checkpoints").is_dir() else []
    return {
        "corpus": (PROCESSED / "corpus.parquet").exists(),
        "features": (PROCESSED / "feat_base.npz").exists(),
        "llm": (PROCESSED / "feat_llm.npz").exists(),
        "trained": (RESULTS / "runs.csv").exists(),
        "robustness": (RESULTS / "robustness.csv").exists(),
        "figures": len(list(FIGURES.glob("fig*.png"))) > 0 if FIGURES.is_dir() else False,
        "extras": (RESULTS / "extra_analyses.json").exists(),
        "triage": (RESULTS / "triage_queue.csv").exists(),
        "checkpoints": len(ckpts) > 0,
    }


def env_report() -> tuple[bool, list[str]]:
    """Check the interpreter can import what the pipeline needs."""
    lines, ok = [], True
    v = sys.version_info
    lines.append(f"Python {v.major}.{v.minor}.{v.micro}")
    if not (3, 9) <= (v.major, v.minor) <= (3, 13):
        lines.append(f"{YELLOW}  (project was written for 3.10-3.12; "
                     f"newer versions may not have PyTorch wheels yet){OFF}")
    try:
        import torch
        backend = "CPU"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            backend = "Apple GPU (MPS)"
        elif torch.cuda.is_available():
            backend = "CUDA"
        lines.append(f"torch {torch.__version__} using {backend}")
    except Exception as e:                                    # noqa: BLE001
        lines.append(f"{RED}torch not importable: {type(e).__name__}: {e}{OFF}")
        ok = False
    try:
        import transformers
        lines.append(f"transformers {transformers.__version__}")
    except Exception as e:                                    # noqa: BLE001
        lines.append(f"{RED}transformers not importable: {e}{OFF}")
        ok = False
    return ok, lines


# --------------------------------------------------------------------------- #
# The menu
# --------------------------------------------------------------------------- #
class Item:
    def __init__(self, key, label, cmd, mins, needs=(), produces=None):
        self.key, self.label, self.cmd = key, label, cmd
        self.mins, self.needs, self.produces = mins, needs, produces

    def blocked_by(self, st: dict) -> str | None:
        for n in self.needs:
            if not st.get(n):
                return n
        return None


NEED_LABEL = {
    "corpus": "the corpus (option 3)",
    "features": "the features (option 4)",
    "llm": "the LLM features (option 5)",
    "trained": "training (option 6)",
    "checkpoints": "a trained checkpoint",
}

ITEMS = [
    Item("1", "Check one image + headline", None, "~1 min", needs=("checkpoints",)),
    Item("2", "Run the tests", [sys.executable, "-m", "pytest", "tests/", "-q"], "~10 sec"),
    Item("3", "Stage 1  Build the corpus", [sys.executable, "-m", "src.build_dataset"],
         "~15 min", produces="corpus"),
    Item("4", "Stage 2  Extract features (CLIP + text)",
         [sys.executable, "-m", "src.features", "--stages", "base"],
         "~30 min", needs=("corpus",), produces="features"),
    Item("5", "Stage 2b Extract LLM features (optional)",
         [sys.executable, "-m", "src.features", "--stages", "llm"],
         "~45 min", needs=("corpus",), produces="llm"),
    Item("6", "Stage 4  Train every model", [sys.executable, "-m", "src.train"],
         "~1-2 hrs", needs=("features",), produces="trained"),
    Item("7", "Stage 6  Robustness + cost", [sys.executable, "-m", "src.robustness"],
         "~15 min", needs=("features",)),
    Item("8", "Stage 5  Figures + significance tests", [sys.executable, "-m", "src.analyse"],
         "~5 min", needs=("trained",)),
    Item("9", "Stage 10 Extended analyses", [sys.executable, "-m", "src.extra_analyses"],
         "~20 min", needs=("trained",)),
    Item("10", "Stage 12 Reviewer triage queue", [sys.executable, "-m", "src.triage"],
         "~1 min", needs=("trained",)),
]

SMOKE = [
    ([sys.executable, "-m", "src.build_dataset", "--limit", "400", "--out", "smoke.parquet"],
     "building a 400-post sample"),
    ([sys.executable, "-m", "src.features", "--corpus", "smoke.parquet",
      "--prefix", "smoke", "--stages", "base"], "encoding it"),
    ([sys.executable, "-m", "src.train", "--corpus", "smoke.parquet", "--prefix", "smoke",
      "--protocols", "random", "--models", "text,concat,cgf", "--seeds", "42",
      "--out", "smoke_runs.csv"], "training three small models"),
]

FULL_SEQUENCE = ["3", "4", "5", "6", "7", "8", "9", "10"]


# --------------------------------------------------------------------------- #
def tick(done: bool) -> str:
    return f"{GREEN}done{OFF}" if done else f"{DIM}not yet{OFF}"


def show_menu(st: dict) -> None:
    print(f"\n{BOLD}{'=' * 62}{OFF}")
    print(f"{BOLD}  Multimodal Misinformation Detection{OFF}")
    print(f"{BOLD}{'=' * 62}{OFF}")

    print(f"\n  {BOLD}Where you are{OFF}")
    for name, key in [("corpus built", "corpus"), ("features extracted", "features"),
                      ("LLM features", "llm"), ("models trained", "trained"),
                      ("figures produced", "figures"),
                      ("trained model saved", "checkpoints")]:
        print(f"    {name:<22} {tick(st[key])}")

    print(f"\n  {BOLD}Try it now{OFF}")
    for it in ITEMS[:2]:
        _print_item(it, st)

    print(f"\n  {BOLD}The pipeline, in order{OFF}")
    for it in ITEMS[2:]:
        _print_item(it, st)

    print(f"\n  {BOLD}Shortcuts{OFF}")
    print(f"    {BLUE}s{OFF}   Quick 5-minute check that everything works")
    print(f"    {BLUE}a{OFF}   Run the whole pipeline start to finish  {DIM}~3 hrs{OFF}")
    print(f"    {BLUE}q{OFF}   Quit")


def _print_item(it: Item, st: dict) -> None:
    blocker = it.blocked_by(st)
    done = it.produces and st.get(it.produces)
    mark = f" {GREEN}(done){OFF}" if done else ""
    if blocker:
        why = NEED_LABEL.get(blocker, blocker)
        print(f"    {DIM}{it.key:<3} {it.label:<38} needs {why}{OFF}")
    else:
        print(f"    {BLUE}{it.key:<3}{OFF} {it.label:<38} {DIM}{it.mins}{OFF}{mark}")


DEFAULT_HEADLINE = "Scientists discover water on the surface of the sun"


def default_image() -> str | None:
    """Any image we can offer as a starting point, so the demo needs no setup."""
    for p in sorted(ROOT.glob("*.jpg")) + sorted(ROOT.glob("*.png")):
        return p.name
    images = ROOT / "data" / "images"
    if images.is_dir():
        for p in sorted(images.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                return str(p.relative_to(ROOT))
    return None


def ask_and_predict() -> bool:
    """Prompt for an image and a headline, then score the pair."""
    suggested = default_image()
    if suggested:
        print(f"\n{DIM}  Press Enter to accept the suggestion in brackets.{OFF}")
        prompt = f"  Image [{suggested}]: "
    else:
        print(f"\n{YELLOW}  No image found in this folder. Type a path to one,{OFF}")
        print(f"{YELLOW}  or drag a file from Finder into this window.{OFF}")
        prompt = "  Image: "

    try:
        image = input(prompt).strip().strip("'\"") or (suggested or "")
        headline = input(f"  Headline [{DEFAULT_HEADLINE}]: ").strip() or DEFAULT_HEADLINE
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if not image:
        print(f"{YELLOW}  No image given.{OFF}")
        return False
    if not (ROOT / image).exists() and not Path(image).exists():
        print(f"{YELLOW}  Cannot find '{image}'. Check the path and try again.{OFF}")
        return False

    return run([sys.executable, "-m", "src.predict",
                "--image", image, "--headline", headline],
               "scoring that image and headline")


def run(cmd: list[str], label: str) -> bool:
    print(f"\n{BOLD}> {label}{OFF}")
    print(f"{DIM}  {' '.join(str(c) for c in cmd)}{OFF}\n")
    t0 = time.time()
    try:
        rc = subprocess.call(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Stopped. Progress so far is cached, so re-running "
              f"picks up where this left off.{OFF}")
        return False
    mins = (time.time() - t0) / 60
    if rc == 0:
        print(f"\n{GREEN}Finished in {mins:.1f} min.{OFF}")
        return True
    print(f"\n{RED}That stage failed (exit code {rc}) after {mins:.1f} min.{OFF}")
    print(f"{DIM}Scroll up for the error. Nothing else was changed.{OFF}")
    return False


def main() -> None:
    print(f"\n{BOLD}Checking your setup...{OFF}")
    ok, lines = env_report()
    for line in lines:
        print(f"  {line}")
    if not ok:
        print(f"\n{RED}The environment is not ready, so nothing here will run.{OFF}")
        print("Fix it with:")
        print(f"  {BOLD}source .venv/bin/activate{OFF}")
        print(f"  {BOLD}pip install -r requirements.txt{OFF}")
        print("If that does not help, the Python version above is the likely cause.")
        return

    by_key = {it.key: it for it in ITEMS}
    while True:
        st = state()
        show_menu(st)
        try:
            choice = input(f"\n  {BOLD}Choose:{OFF} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if choice in ("q", "quit", "exit"):
            print("Bye.")
            return

        if choice == "s":
            print(f"\n{BOLD}Quick check: a miniature run of the whole path.{OFF}")
            for cmd, label in SMOKE:
                if not run(cmd, label):
                    break
            else:
                print(f"\n{GREEN}Everything works. You are ready for the real run.{OFF}")
            continue

        if choice == "a":
            print(f"\n{YELLOW}This runs every stage back to back and takes "
                  f"around three hours.{OFF}")
            if input("  Type yes to start: ").strip().lower() != "yes":
                continue
            for key in FULL_SEQUENCE:
                it = by_key[key]
                if it.produces and state().get(it.produces):
                    print(f"{DIM}Skipping {it.label} (already done).{OFF}")
                    continue
                if not run(it.cmd, it.label):
                    break
            else:
                print(f"\n{GREEN}Pipeline complete. Figures are in figures/, "
                      f"numbers in results/.{OFF}")
            continue

        it = by_key.get(choice)
        if it is None:
            print(f"{YELLOW}  '{choice}' is not on the menu.{OFF}")
            continue
        blocker = it.blocked_by(state())
        if blocker:
            print(f"{YELLOW}  Run {NEED_LABEL.get(blocker, blocker)} first.{OFF}")
            continue
        if it.cmd is None:
            ask_and_predict()
        else:
            run(it.cmd, it.label)


if __name__ == "__main__":
    main()
