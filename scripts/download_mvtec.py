"""Fetch the MVTec AD 'toothbrush' category from the Hugging Face mirror.

Same source VisionAI Project 2 used (verified 2026-10-21):
    https://huggingface.co/datasets/foersben/mvtec-ad  (CC BY-NC-SA 4.0,
    research / non-commercial)

Downloads the 134 files under toothbrush/ into data_raw/mvtec/toothbrush/.
"""
import sys
import urllib.request
from pathlib import Path

MIRROR = "https://huggingface.co/datasets/foersben/mvtec-ad/resolve/main"
CATEGORY = "toothbrush"
# fixed file list captured from the HF Hub API listing (2026-10-21)
SUBDIRS = {
    "train/good": 60,
    "test/good": 12,
    "test/defective": 30,
    "ground_truth/defective": 30,
}


def main() -> None:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = [f.rpath for f in api.list_repo_files("foersben/mvtec-ad", repo_type="dataset")
                 if f.rpath.startswith(f"{CATEGORY}/")]
    except Exception as e:  # pragma: no cover - network path
        print(f"HF Hub listing failed ({e}); install huggingface_hub or set FORGEAD_DATA.")
        sys.exit(1)

    dest = Path(__file__).resolve().parent.parent / "data_raw" / "mvtec" / CATEGORY
    ok, fail = 0, 0
    for rel in sorted(files):
        target = dest / rel[len(CATEGORY) + 1:]
        if target.exists():
            ok += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(f"{MIRROR}/{rel}", target)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"FAILED {rel}: {e}")
        if (ok + fail) % 20 == 0:
            print(f"... {ok + fail}/{len(files)}")
    print(f"done: {ok} ok, {fail} failed -> {dest}")


if __name__ == "__main__":
    main()
