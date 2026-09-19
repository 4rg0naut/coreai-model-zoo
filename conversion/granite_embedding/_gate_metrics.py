"""One finite, shape-exact numerical + retrieval gate shared by every stage (torch, Mac runtime, iPhone)."""
import numpy as np

COSINE_MIN = 0.999
MAX_ABS = 0.02
NORM_ERROR_MAX = 0.002
SIMILARITY_ERROR_MAX = 0.01


def tensor_metrics(reference, candidate):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        return {"finite": False, "shape_match": False, "reference_shape": list(a.shape), "candidate_shape": list(b.shape)}
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"finite": False, "shape_match": True}
    # Avoid BLAS state/FP-flag effects after inference; accumulate metrics explicitly.
    na, nb = float(np.sqrt(np.sum(a*a))), float(np.sqrt(np.sum(b*b)))
    cosine = float(np.sum(a*b) / (na*nb)) if na*nb else 0.0
    return {"finite": bool(np.isfinite([na,nb,cosine]).all()), "shape_match": True, "cosine": cosine,
            "max_abs": float(np.max(np.abs(a-b))), "rmse": float(np.sqrt(np.mean((a-b)**2))),
            "reference_norm": na, "candidate_norm": nb, "norm_error": abs(na-nb)}


def gate_vectors(golden, vectors):
    fixtures = golden["fixtures"]
    ids = [r["id"] for r in fixtures]
    if not fixtures or len(ids) != len(set(ids)) or not any(r["kind"] == "query" for r in fixtures) or sum(r["kind"] == "document" for r in fixtures) < 2:
        return {"status": "FAIL", "failures": ["empty/duplicate fixtures or missing retrieval coverage"],
                "per_fixture": {}, "retrieval": [], "min_cosine": 0.0, "max_abs": 1e30}
    reference = {r["id"]: np.asarray(r["embedding"], dtype=np.float64) for r in fixtures}
    failures, per_fixture = [], {}
    if set(reference) != set(vectors):
        failures.append("fixture IDs differ")
    for key, a in reference.items():
        m = tensor_metrics(a, vectors.get(key, []))
        per_fixture[key] = m
        if not m["finite"] or not m["shape_match"] or m.get("cosine", 0) < COSINE_MIN or m.get("max_abs", float("inf")) > MAX_ABS or m.get("norm_error", float("inf")) > NORM_ERROR_MAX:
            failures.append(f"numerical:{key}")
    retrieval = []
    if all(m["finite"] and m["shape_match"] for m in per_fixture.values()):
        docs = [r["id"] for r in fixtures if r["kind"] == "document"]
        queries = [r["id"] for r in fixtures if r["kind"] == "query"]
        for query in queries:
            a = np.asarray([np.sum(reference[query]*reference[d]) for d in docs])
            b = np.asarray([np.sum(np.asarray(vectors[query])*np.asarray(vectors[d])) for d in docs])
            ao, bo = np.argsort(-a, kind="stable"), np.argsort(-b, kind="stable")
            margin_a, margin_b = float(a[ao[0]]-a[ao[1]]), float(b[bo[0]]-b[bo[1]])
            clear_pair_flips = sum((a[i]-a[j]) * (b[i]-b[j]) <= 0 and abs(a[i]-a[j]) >= 0.001 for i in range(len(docs)) for j in range(i))
            entry = {"query": query, "reference_order": [docs[i] for i in ao], "candidate_order": [docs[i] for i in bo],
                     "reference_scores": a.tolist(), "candidate_scores": b.tolist(), "reference_margin": margin_a,
                     "candidate_margin": margin_b, "max_score_error": float(np.max(abs(a-b))),
                     "top1_match": bool(ao[0] == bo[0]), "clear_pair_flips": int(clear_pair_flips)}
            if not entry["top1_match"] or entry["max_score_error"] > SIMILARITY_ERROR_MAX or clear_pair_flips:
                failures.append(f"retrieval:{query}")
            retrieval.append(entry)
    return {"status": "FAIL" if failures else "PASS", "thresholds": {"cosine_min": COSINE_MIN, "max_abs": MAX_ABS,
                "norm_error_max": NORM_ERROR_MAX, "similarity_error_max": SIMILARITY_ERROR_MAX,
                "top1": "exact for every query", "pairwise_order": "exact when oracle gap >=0.001"},
            "failures": failures, "per_fixture": per_fixture, "retrieval": retrieval,
            "min_cosine": min((m.get("cosine", 0) for m in per_fixture.values()), default=0),
            "max_abs": max((m.get("max_abs", 1e30) for m in per_fixture.values()), default=1e30)}
