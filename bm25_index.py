# ============================================================
# Module: BM25 Keyword Index (bm25_index.py)
# 模塊：BM25 關鍵詞索引 —— 摺好的第二把降落傘
#
# Okapi BM25 over bucket text with a built-in tokenizer:
# English/digit runs as words, CJK runs as bigrams (+ unigrams for
# single-char coverage). Pure stdlib — no jieba, no external deps.
#
# Dormant by default: wired into bucket_manager.search() behind the
# config flag matching.bm25_enabled (default False). Trigger to flip:
# the corpus outgrows rapidfuzz (2000+ buckets) or real recall misses.
# 自寫 Okapi BM25，內建斷詞（英數連段＋中日韓 bigram/unigram），零依賴。
# 預設休眠：由 matching.bm25_enabled 旗標控制，觸發條件到了改一行 config。
#
# Depended on by: bucket_manager.py
# ============================================================

from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[a-z0-9_]+")
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]+")


def tokenize(text: str, for_query: bool = False) -> list[str]:
    """English/digit runs as word tokens; CJK runs as bigrams.
    The index side also emits unigrams so single-character queries still hit;
    the query side prefers bigrams and falls back to unigrams only when a run
    is a single character (unigram queries against a bigram-only index miss).
    索引端同時吐 unigram（讓單字查詢有得比），查詢端以 bigram 為主。"""
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = _WORD_RE.findall(lowered)
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
        if not for_query:
            tokens.extend(run)  # unigrams, index side only
    return tokens


class Bm25Index:
    """In-memory Okapi BM25 index, rebuilt when the corpus version changes.
    At our scale (hundreds of buckets) a rebuild is tens of milliseconds."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.version = None            # opaque corpus-version key set by build()
        self._doc_freqs: dict[str, Counter] = {}
        self._doc_lens: dict[str, int] = {}
        self._df: Counter = Counter()  # term → number of docs containing it
        self._avg_len = 0.0

    def build(self, docs: dict[str, str], version=None) -> None:
        """docs: {doc_id: text}. version: any hashable corpus fingerprint."""
        self._doc_freqs = {}
        self._doc_lens = {}
        self._df = Counter()
        for doc_id, text in docs.items():
            tokens = tokenize(text)
            freqs = Counter(tokens)
            self._doc_freqs[doc_id] = freqs
            self._doc_lens[doc_id] = len(tokens)
            for term in freqs:
                self._df[term] += 1
        n = len(self._doc_lens)
        self._avg_len = (sum(self._doc_lens.values()) / n) if n else 0.0
        self.version = version

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Returns [(doc_id, score)] sorted desc; empty when nothing matches."""
        q_terms = tokenize(query, for_query=True)
        if not q_terms or not self._doc_freqs:
            return []
        n = len(self._doc_freqs)
        scores: dict[str, float] = {}
        for term in q_terms:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, freqs in self._doc_freqs.items():
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_lens[doc_id] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_len)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]
