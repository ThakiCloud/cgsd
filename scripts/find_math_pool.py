#!/usr/bin/env python3
"""집 머신에서 한 번 돌리는 Math/SQL 게이트 데이터 탐색·검증기 (v1.13 D83).

P5(cross-domain)에 필요한 것은 ESCI 풀과 **같은 모양**이다. 이 스크립트는 집 머신을 훑어
후보 파일을 찾고, CGSD 가 실제로 쓸 수 있는지 필드 단위로 판정한다. 아무것도 고치지 않고
읽기만 한다.

    python3 find_math_pool.py                 # 홈 전체 탐색
    python3 find_math_pool.py --root ~/some   # 범위 지정

필요 조건 (하나라도 없으면 그 파일은 사용 불가):
  query   pre-gate 원문(게이트가 만들지 않은 입력)
  label   학습 타깃
  gate    게이트 판정 (pass/fail, 또는 map_gate+tr_gate 처럼 두 축)
  tag     support 후보 — ⛔ 게이트 산출물이면 실격 (v1.2 D10/D11 의 ADF leaf 와 같은 함정)
"""
import argparse, json, os, re, sys
from pathlib import Path

HINT = re.compile(r"(math|sql|gsm|reason|tool|code)", re.I)
SKIP = re.compile(r"(node_modules|site-packages|\.git/|\.venv|Library/|__pycache__|/Caches/)")
QUERY = ("query", "question", "problem", "prompt", "input", "nl", "utterance")
LABEL = ("label", "answer", "solution", "leaf", "target", "sql", "output", "completion")
GATE = ("gate", "pass", "verdict", "accepted", "map_gate", "tr_gate", "is_valid", "filtered")
TAG = ("tag", "topic", "category", "skill", "op", "type", "domain", "subject")


def peek(p, n=3):
    rows = []
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if len(rows) >= n:
                    break
    except Exception:
        return []
    return [r for r in rows if isinstance(r, dict)]


def classify(keys):
    low = {k.lower(): k for k in keys}
    hit = lambda names: next((low[k] for k in low if any(n in k for n in names)), None)
    return {"query": hit(QUERY), "label": hit(LABEL), "gate": hit(GATE), "tag": hit(TAG)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home()))
    ap.add_argument("--max-files", type=int, default=400)
    a = ap.parse_args()
    cands, seen = [], 0
    for dirpath, dirnames, files in os.walk(a.root):
        if SKIP.search(dirpath + "/"):
            dirnames[:] = []
            continue
        for fn in files:
            if not fn.endswith((".jsonl", ".json")):
                continue
            p = Path(dirpath) / fn
            if not (HINT.search(str(p)) or HINT.search(fn)):
                continue
            seen += 1
            if seen > a.max_files:
                break
            rows = peek(p)
            if not rows:
                continue
            f = classify(rows[0].keys())
            if f["query"] and f["label"]:
                try:
                    n = sum(1 for _ in open(p, encoding="utf-8", errors="ignore"))
                except Exception:
                    n = -1
                cands.append((p, f, n, sorted(rows[0].keys())[:12]))
    if not cands:
        print("후보 없음 — --root 를 좁혀 다시 돌리거나, 데이터가 다른 이름/포맷일 수 있다.")
        return 1
    cands.sort(key=lambda c: (c[1]["gate"] is None, -c[2]))
    print(f"후보 {len(cands)}개 (게이트 판정이 있는 것부터)\n")
    for p, f, n, keys in cands[:15]:
        ok = "✅ 사용 가능" if f["gate"] else "⚠️ 게이트 판정 필드 없음"
        print(f"{ok}  {p}  rows={n:,}")
        print(f"   query={f['query']}  label={f['label']}  gate={f['gate']}  tag={f['tag']}")
        print(f"   keys: {', '.join(keys)}")
        if f["tag"]:
            print(f"   ⛔ 확인 필요: '{f['tag']}' 가 게이트 산출물이면 support 로 쓸 수 없다 (v1.2 D10/D11)")
        print()
    print("다음 단계: ✅ 인 파일 경로를 알려주거나, 그 파일 + 홀드아웃 + 기존 15 seed 결과를\n"
          "          tar 로 묶어 이 레포가 접근 가능한 곳으로 옮긴다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
