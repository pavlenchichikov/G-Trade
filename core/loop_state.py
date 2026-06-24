"""Read/write helpers for loop_state.json, the single source of truth the loop
cycle writes and the webUI reads. Kept tiny and file-based so it is trivial to
test and inspect."""
import json
import os


def _skeleton():
    return {"last_run": None, "steps": {}, "assets": [],
            "proposed": [], "approved": [], "history": []}


def load_state(path):
    if not os.path.exists(path):
        return _skeleton()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return _skeleton()
    skel = _skeleton()
    skel.update(data)
    return skel


def save_state(path, state):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def approve(path, assets):
    st = load_state(path)
    proposed = set(st.get("proposed", []))
    keep = set(st.get("approved", []))
    keep.update(a for a in assets if a in proposed)
    st["approved"] = sorted(keep)
    save_state(path, st)
    return st


def dismiss(path, asset):
    st = load_state(path)
    st["proposed"] = [a for a in st.get("proposed", []) if a != asset]
    save_state(path, st)
    return st
