"""Sinh cặp genuine / impostor (mục 8.2)."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .datasets import Record


@dataclass(frozen=True)
class Pair:
    probe: Record
    ref: Record
    label: int  # 1 = genuine, 0 = impostor
    same_subject: bool  # impostor cùng người, khác ngón


def own_sets(records: list[Record], cond: str, subjects: set[str] | None = None
             ) -> tuple[list[Record], list[Record]]:
    """Enroll luôn là s1_c; probe là s2_{cond}."""
    keep = (lambda r: True) if subjects is None else (lambda r: r.subject in subjects)
    enroll = [r for r in records if r.session == "s1" and r.cond == "c" and keep(r)]
    probes = [r for r in records if r.session == "s2" and r.cond == cond and keep(r)]
    return enroll, probes


def own_pairs(records: list[Record], cond: str, subjects: set[str] | None = None) -> list[Pair]:
    """Genuine: (enroll s1_c, probe s2_cond) cùng (subject, finger). Impostor: khác ngón."""
    enroll, probes = own_sets(records, cond, subjects)
    pairs = []
    for p in probes:
        for e in enroll:
            genuine = p.identity == e.identity
            pairs.append(Pair(p, e, int(genuine),
                              (not genuine) and p.subject == e.subject))
    return pairs


def fvc_pairs(records: list[Record]) -> list[Pair]:
    """FVC: genuine = mọi cặp mẫu cùng ngón; impostor = mẫu đầu tiên của các ngón khác nhau."""
    by_id: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_id[r.identity].append(r)
    ids = sorted(by_id)
    pairs = []
    for i in ids:
        rs = sorted(by_id[i], key=lambda r: r.sample)
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                pairs.append(Pair(rs[b], rs[a], 1, False))
    firsts = [min(by_id[i], key=lambda r: r.sample) for i in ids]
    for a in range(len(firsts)):
        for b in range(a + 1, len(firsts)):
            pairs.append(Pair(firsts[b], firsts[a], 0, False))
    return pairs


def write_pairs_csv(pairs: list[Pair], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["probe", "ref", "label", "same_subject"])
        for p in pairs:
            w.writerow([p.probe.path, p.ref.path, p.label, int(p.same_subject)])
