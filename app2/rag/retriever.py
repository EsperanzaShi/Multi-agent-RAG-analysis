from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("data/index")

KW_POS = [
    "climate", "climate change", "greenhouse gas", "ghg", "emission", "emissions",
    "transition risk", "physical risk", "severe weather", "carbon", "decarbon",
    "methane", "net zero", "tcfd", "resilience", "adaptation", "carbon price",
    "carbon pricing", "carbon tax", "regulatory risk", "regulation"
]
KW_NEG = ["internal control", "controls and procedures", "icfr", "item 9a"]

def _kw_score(text: str) -> float:
    t = text.lower()
    pos = sum(1 for kw in KW_POS if kw in t)
    neg = sum(1 for kw in KW_NEG if kw in t)
    # Increase the positive impact of climate keywords and reduce the negative weight
    # so we keep climate-relevant pages even when Top-K is small.
    return 0.6 * pos - 0.3 * neg

class Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
        self.meta: List[Dict[str, Any]] = []
        with open(INDEX_DIR / "meta.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # quick lookup: (doc, page) -> list of meta entries (text/table chunks)
        self.by_doc_page: Dict[Tuple[str,int], List[Dict[str,Any]]] = {}
        for m in self.meta:
            key = (m.get("doc"), int(m.get("page_start", 0)))
            self.by_doc_page.setdefault(key, []).append(m) #Group chunks by document+page
        # Stats for last search (used for UI tracing)
        self.stats: Dict[str, Any] = {}

    def _neighbors(self, m: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        doc = m.get("doc")
        p = int(m.get("page_start", 0))
        # Expand neighbor radius to capture nearby context
        for dp in (-2, -1, 1, 2):
            for mm in self.by_doc_page.get((doc, p + dp), []):
                out.append(mm)
        return out

    def search(
        self,
        query: str,
        top_k: int = 6,
        company: Optional[str] = None,
        years: Optional[List[int]] = None,
        item_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        import time
        t0 = time.time()
        # Only expand query with climate terms if the original query is about climate/risk
        if any(term in query.lower() for term in ["climate", "risk", "emission", "greenhouse", "carbon", "environmental"]):
            expanded_query = query + " climate climate change greenhouse gas GHG emissions TCFD transition risk physical risk carbon decarbonization"
        else:
            expanded_query = query
        q = self.model.encode([expanded_query], normalize_embeddings=True).astype("float32")

        # Search a larger pool to improve recall; we will re-rank with domain-specific features
        pool = min(1000, self.index.ntotal)
        # FAISS returns the top 1000 most similar chunks based on embedding similarity
        D, I = self.index.search(q, pool)

        candidates: List[Tuple[float, Dict[str, Any]]] = []
        seen_ids = set()

        def add_candidate(score: float, m: Dict[str,Any], bonus: float = 0.0):
            mid = m.get("id")
            if mid in seen_ids:
                return
            seen_ids.add(mid)
            candidates.append((score + bonus, m))

        for idx, sim in zip(I[0], D[0]):
            if idx == -1:
                continue
            m = self.meta[idx]

            # metadata filters
            if company and m.get("company") and company.lower() not in m["company"].lower():
                continue
            if years and m.get("year") not in years:
                continue

            score = float(sim) + _kw_score(m.get("text", ""))

            item_val = (m.get("item") or "").lower()
            if "item 1a" in item_val or "risk factor" in m.get("text", "").lower():
                score += 0.6
            if item_hint and item_val and item_hint.lower() in item_val:
                score += 0.35

            add_candidate(score, m)

            # Main chunk ranks higher, but neighbors still provide context.
            for nb in self._neighbors(m):
                add_candidate(score - 0.02, nb)

        # sort by composite score and return unique top_k
        #For each tuple x, return the first element x[0] (the score)
        candidates.sort(key=lambda x: x[0], reverse=True)
        results: List[Dict[str, Any]] = []
        seen_pages = set()
        for _, m in candidates:
            key = (m.get("doc"), m.get("page_start"))
            #Provides broader coverage across different pages
            if key in seen_pages:
                continue
            results.append(m)
            seen_pages.add(key)
            if len(results) >= max(top_k, 1):
                break
        # Backfill if we still have fewer than top_k results: select highest KW score pages
        if len(results) < top_k:
            filtered_meta = [m for m in self.meta
                             if (not company or (m.get("company") and company.lower() in m["company"].lower()))
                             and (not years or m.get("year") in years)]
            # Sort by keyword score primarily, then by whether text mentions risk explicitly
            def backfill_score(m: Dict[str, Any]) -> float:
                base = _kw_score(m.get("text", ""))
                if "risk" in (m.get("text", "").lower()):
                    base += 0.3
                return base
            for m in sorted(filtered_meta, key=backfill_score, reverse=True):
                key = (m.get("doc"), m.get("page_start"))
                if key in seen_pages:
                    continue
                results.append(m)
                seen_pages.add(key)
                if len(results) >= top_k:
                    break
        # Record stats
        self.stats = {
            "query": query,
            "expanded": expanded_query != query,
            "pool": pool,
            "candidates_considered": len(candidates),
            "top_k": top_k,
            "results": len(results),
            "time_s": round(time.time() - t0, 3),
        }
        return results