"""Fuzzy-match track_label.txt entries against the analyzed library in db/djia.db
and (re)write results/track_labels_matched.csv.

Each track_label.txt line is "<name> - <label(s)>". For each name we score every track
in the DB by fuzzy partial-ratio (typo-tolerant) against its file_name and title, then
keep every track within a small tolerance of the best score (so genuine duplicate files
in the library all show up, as in the original matched CSV).
"""
import csv
import difflib
import re
import sqlite3
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LABEL_FILE = "track_label.txt"
DB_PATH = "db/djia.db"
OUT_CSV = "results/track_labels_matched.csv"

BEST_SCORE_THRESHOLD = 0.80   # below this, treat as "no match"
TIE_TOLERANCE = 0.02          # keep every candidate within this of the best score
MAX_MATCHES_PER_NAME = 4      # flag (don't silently dump) names with more ties than this


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_score(name_tokens, cand_tokens) -> float:
    """Average, per name token, of its best fuzzy match among candidate tokens.
    Word-level (not raw character substring) so short names like 'io' don't spuriously
    match inside unrelated words like 'audio' or 'studio'."""
    if not name_tokens or not cand_tokens:
        return 0.0
    total = 0.0
    for nt in name_tokens:
        best = max(difflib.SequenceMatcher(None, nt, ct).ratio() for ct in cand_tokens)
        total += best
    return total / len(name_tokens)


def parse_labels(path: str):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, _, label = line.rpartition(" - ")
            if not name:
                continue
            entries.append((name.strip(), label.strip()))
    return entries


def load_tracks(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT id, file_name, title FROM tracks ORDER BY id")
    tracks = [dict(r) for r in cur.fetchall()]
    conn.close()
    for t in tracks:
        t["_file_tokens"] = normalize(t["file_name"] or "").split()
        t["_title_tokens"] = normalize(t["title"] or "").split()
    return tracks


def best_matches(name: str, tracks):
    name_tokens = normalize(name).split()
    scored = []
    for t in tracks:
        score = token_score(name_tokens, t["_file_tokens"])
        if t["_title_tokens"]:
            score = max(score, token_score(name_tokens, t["_title_tokens"]))
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] < BEST_SCORE_THRESHOLD:
        return []
    top = scored[0][0]
    return [t for score, t in scored if score >= top - TIE_TOLERANCE]


def main():
    entries = parse_labels(LABEL_FILE)
    tracks = load_tracks(DB_PATH)
    print(f"{len(entries)} label entries, {len(tracks)} tracks in DB")

    rows = []
    unmatched = []
    flagged = []  # (name, count) - too many ties, likely an over-generic name
    for name, label in entries:
        matches = best_matches(name, tracks)
        if not matches:
            unmatched.append(name)
            continue
        if len(matches) > MAX_MATCHES_PER_NAME:
            flagged.append((name, len(matches)))
        for t in matches:
            rows.append({"track_file": t["file_name"], "track_name": name, "label": label})

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_file", "track_name", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} matched rows -> {OUT_CSV}")
    if unmatched:
        print(f"\n{len(unmatched)} unmatched entries:")
        for name in unmatched:
            print(f"  - {name}")
    if flagged:
        print(f"\n{len(flagged)} names with more than {MAX_MATCHES_PER_NAME} tied matches "
              "(likely too generic - review manually):")
        for name, count in flagged:
            print(f"  - {name!r}: {count} matches")


if __name__ == "__main__":
    main()
