"""Evaluation v7 (graph mode as main, classic as reference)."""
import argparse, asyncio, ast, csv, json, math, os, re, sys, time, warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]          # lm_service
TAGS = ROOT / "eval" / "magisterka_eval" / "tags"
TAG = os.environ.get("EVAL_TAG", "v7")
OUT = TAGS / TAG
TESTSET_PATH = TAGS / "testset_clean.csv"
QA_ENDPOINT = "http://localhost:8080/qa"
JUDGE_MODEL = "gpt-4o-mini"
METRIC_COLS = ["faithfulness", "factual_correctness(mode=f1)", "response_groundedness", "context_relevance", "answer_accuracy"]
EXTRA_COLS = ["factual_correctness(mode=precision)", "factual_correctness(mode=recall)"]
IDK = "nie znam odpowiedzi"


def load_openai_key() -> None:
    key_file = Path(os.environ.get("OPENAI_KEY_FILE", ""))
    if key_file.is_file():
        raw = key_file.read_text(encoding="utf-8-sig").strip()
        raw = "".join(ch for ch in raw if ch.isprintable() and not ch.isspace())
        if "=" in raw:
            raw = raw.split("=", 1)[1].strip().strip('"').strip("'")
        if raw:
            os.environ["OPENAI_API_KEY"] = raw


def load_testset():
    import pandas as pd
    df = pd.read_csv(TESTSET_PATH, encoding="utf-8-sig")
    rows = []
    for _, r in df.iterrows():
        q = str(r["user_input"]).strip()
        if q:
            rows.append({"user_input": q, "reference": str(r["reference"]).strip(), "reference_contexts": r["reference_contexts"]})
    return rows


# ----------------------------------------------------------------------------- collect
async def _call_qa(client, question: str, rag_mode: str) -> Dict:
    resp = await client.post(QA_ENDPOINT, json={"question": question}, params={"rag_mode": rag_mode})
    resp.raise_for_status()
    data = resp.json()
    return {"response": data.get("answer") or "", "retrieved_contexts": data.get("source_context") or "",
            "reasoning": data.get("reasoning") or ""}


async def collect(modes: List[str], runs: int, concurrency: int, limit: int | None):
    import httpx, pandas as pd
    samples = load_testset()[:limit] if limit else load_testset()
    OUT.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=30.0)) as client:
        for run in range(1, runs + 1):
            path = OUT / f"qa_results_{TAG}_run{run}.csv"
            done = set()
            rows = []
            if path.exists():
                old = pd.read_csv(path, encoding="utf-8-sig")
                rows = old.to_dict("records")
                done = {(r["mode"], r["question"]) for r in rows if str(r.get("response") or "").strip()}
            todo = [(m, s) for m in modes for s in samples if (m, s["user_input"]) not in done]
            print(f"[run {run}] to collect: {len(todo)} (done: {len(done)})", flush=True)
            t0 = time.time()

            async def one(mode, s):
                async with sem:
                    for attempt in range(3):
                        try:
                            out = await _call_qa(client, s["user_input"], mode)
                            break
                        except Exception as e:
                            print(f"  [{mode}] error ({attempt + 1}/3) for '{s['user_input'][:50]}': {e}", flush=True)
                            await asyncio.sleep(5)
                    else:
                        out = {"response": "", "retrieved_contexts": "", "reasoning": ""}
                    return {"mode": mode, "question": s["user_input"], "reasoning": out["reasoning"],
                            "response": out["response"], "retrieved_contexts": out["retrieved_contexts"], "run": run}

            results = []
            for i in range(0, len(todo), concurrency * 4):
                chunk = todo[i:i + concurrency * 4]
                results += await asyncio.gather(*[one(m, s) for m, s in chunk])
                pd.DataFrame(rows + results).to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
                print(f"  [{min(i + len(chunk), len(todo))}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows + results).to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
            df = pd.DataFrame(rows + results)
            for m in modes:
                d = df[df["mode"] == m]
                idk = d["response"].fillna("").str.strip().str.lower().str.startswith(IDK).sum()
                empty = (d["response"].fillna("").str.strip() == "").sum()
                print(f"  run {run} {m}: n={len(d)} refusals={idk} empty={empty}", flush=True)


# ----------------------------------------------------------------------------- score
def _safe_ctx(val) -> List[str]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ["brak kontekstu"]
    s = str(val).strip()
    if not s:
        return ["brak kontekstu"]
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list) and parsed:
                return [str(x) for x in parsed]
        except Exception:
            pass
    return [s]


def score(runs: List[int], batch: int):
    import pandas as pd
    from ragas import evaluate, EvaluationDataset, RunConfig
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, FactualCorrectness, ResponseGroundedness, ContextRelevance, AnswerAccuracy
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    judge = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, max_tokens=16384))
    metrics = [
        (Faithfulness(llm=judge), "faithfulness"),
        (FactualCorrectness(llm=judge), "factual_correctness(mode=f1)"),
        (ResponseGroundedness(llm=judge), "response_groundedness"),
        (ContextRelevance(llm=judge), "context_relevance"),
        (AnswerAccuracy(llm=judge), "answer_accuracy"),
        (FactualCorrectness(llm=judge, mode="precision"), "factual_correctness(mode=precision)"),
        (FactualCorrectness(llm=judge, mode="recall"), "factual_correctness(mode=recall)"),
    ]
    ref = pd.DataFrame(load_testset())
    for run in runs:
        src = OUT / f"qa_results_{TAG}_run{run}.csv"
        dst = OUT / f"merged_final_{TAG}_run{run}.csv"
        raw = pd.read_csv(src, encoding="utf-8-sig").rename(columns={"question": "user_input"})
        raw = raw.drop_duplicates(subset=["user_input", "mode"]).merge(ref, on="user_input", how="left")
        for c in METRIC_COLS + EXTRA_COLS:
            raw[c] = None
        raw = raw.reindex(columns=["user_input", "retrieved_contexts", "reference_contexts", "response", "reasoning",
                                   "reference", "mode"] + METRIC_COLS + EXTRA_COLS + ["run"])
        if dst.exists():
            df = pd.read_csv(dst, encoding="utf-8-sig")
            for c in METRIC_COLS + EXTRA_COLS:
                if c not in df.columns:
                    df[c] = None
            have = set(zip(df["mode"], df["user_input"]))
            extra = raw[[(m, q) not in have for m, q in zip(raw["mode"], raw["user_input"])]]
            if len(extra):
                print(f"[run {run}] appending {len(extra)} new rows (modes: {sorted(extra['mode'].unique())})", flush=True)
                df = pd.concat([df, extra], ignore_index=True)
        else:
            df = raw
        for metric, col in metrics:
            missing = df[df[col].isnull()].index.tolist()
            print(f"[run {run}] {col}: to compute {len(missing)}", flush=True)
            for i in range(0, len(missing), batch):
                idx = missing[i:i + batch]
                samples = [SingleTurnSample(user_input=str(df.at[j, "user_input"]),
                                            response=str(df.at[j, "response"]).strip() or "brak odpowiedzi",
                                            reference=str(df.at[j, "reference"]),
                                            retrieved_contexts=_safe_ctx(df.at[j, "retrieved_contexts"]),
                                            reference_contexts=_safe_ctx(df.at[j, "reference_contexts"])) for j in idx]
                try:
                    res = evaluate(dataset=EvaluationDataset(samples=samples), metrics=[metric],
                                   run_config=RunConfig(max_workers=8, timeout=240, max_retries=3), show_progress=False)
                    rdf = res.to_pandas()
                    rcol = metric.name if metric.name in rdf.columns else [c for c in rdf.columns if c.startswith(metric.name.split("(")[0])][0]
                    for j, v in zip(idx, rdf[rcol].tolist()):
                        df.at[j, col] = None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
                except Exception as e:
                    print(f"  batch error {i}: {e}", flush=True)
                df.to_csv(dst, index=False, encoding="utf-8-sig")
                print(f"  {col}: {min(i + batch, len(missing))}/{len(missing)}", flush=True)
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"[run {run}] saved {dst}", flush=True)


# ----------------------------------------------------------------------------- graph metrics (offline)
def _graph_dump():
    """Dump of the Neo4j graph and the chunk -> document mapping from Qdrant (vectorizer venv is not needed: neo4j and qdrant_client are in the lm venv?)"""
    from neo4j import GraphDatabase
    import httpx
    d = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", os.environ["NEO4J_PASSWORD"]))
    with d.session() as s:
        nodes = s.run("MATCH (n:Node) RETURN n.name AS name, n.chunk_ids AS cids").data()
        rels = s.run("MATCH (h:Node)-[r:RELATED]->(t:Node) RETURN h.name AS h, r.type AS rt, t.name AS t, r.chunk_ids AS cids").data()
    d.close()
    # Qdrant REST scroll (without the gRPC client)
    chunk2doc = {}
    nxt = None
    with httpx.Client(timeout=60) as c:
        while True:
            body = {"limit": 500, "with_payload": True, "with_vector": False}
            if nxt:
                body["offset"] = nxt
            r = c.post("http://localhost:6333/collections/vectorstore/points/scroll", json=body).json()["result"]
            for p in r["points"]:
                md = p["payload"].get("metadata", {})
                chunk2doc[md.get("chunk_id")] = md.get("document_name")
            nxt = r.get("next_page_offset")
            if not nxt:
                break
    return nodes, rels, chunk2doc


def graphmetrics(files: List[str]):
    import pandas as pd, numpy as np
    nodes, rels, chunk2doc = _graph_dump()
    doc2art = {d: (re.match(r"(\d{7})", d).group(1) if d and re.match(r"(\d{7})", d) else d) for d in set(chunk2doc.values())}
    art_nodes, art_rels = {}, {}
    for n in nodes:
        for c in (n["cids"] or []):
            a = doc2art.get(chunk2doc.get(c))
            if a:
                art_nodes.setdefault(a, set()).add(n["name"])
    for r in rels:
        for c in (r["cids"] or []):
            a = doc2art.get(chunk2doc.get(c))
            if a:
                art_rels.setdefault(a, set()).add((r["h"], r["rt"], r["t"]))

    def parse_ctx(c):
        c = str(c)
        hits = re.findall(r"(?:Znaleziony Byt \d+: |^- )(.+?) \(Typ: ", c, flags=re.M)
        trip = re.findall(r"\((?:[^:()]+): ([^)]+?)\) --\[([^\]]+)\]--> \((?:[^:()]+): ([^)]+?)\)", c)
        return hits, trip

    def nums(s):
        return {x for x in re.findall(r"\d+(?:[.,]\d+)?", str(s)) if len(x) > 1}

    def related(h, q):
        h = h.lower()
        return h in q or any(tok in q for tok in h.split() if len(tok) > 4)

    out = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        qcol = "user_input" if "user_input" in df.columns else "question"
        g = df[df["mode"] == "graph"].copy()
        if "reference_contexts" not in g.columns:
            g = g.merge(pd.DataFrame(load_testset()), left_on=qcol, right_on="user_input", how="left")
        stats = dict(file=Path(f).name, n=len(g))
        seed_rel, trip_art, num_ctx, num_graph, n_trip, reach = [], [], [], [], [], []
        for _, r in g.iterrows():
            q = str(r[qcol]).lower()
            hits, trip = parse_ctx(r["retrieved_contexts"])
            if hits:
                seed_rel.append(sum(1 for h in hits if related(h, q)) / len(hits))
            arts = set(re.findall(r"\b(1\d{6})\b", str(r.get("reference_contexts", ""))))
            n_trip.append(len(trip))
            if len(arts) == 1:
                a = next(iter(arts))
                art_t = art_rels.get(a, set())
                names = set(hits) | {h for h, _, t in trip} | {t for h, _, t in trip}
                reach.append(1.0 if names & art_nodes.get(a, set()) else 0.0)
                if trip:
                    trip_art.append(sum(1 for h, rt, t in trip if (h, rt, t) in art_t or (t, rt, h) in art_t) / len(trip))
                rn = nums(r.get("reference", ""))
                if rn:
                    ctx = str(r["retrieved_contexts"])
                    num_ctx.append(sum(1 for x in rn if x in ctx) / len(rn))
                    num_graph.append(sum(1 for x in rn if x in " ".join(art_nodes.get(a, set()))) / len(rn))
        stats.update(seed_relevance=np.mean(seed_rel) if seed_rel else float("nan"),
                     triples_in_ctx=np.mean(n_trip),
                     share_triples_from_ref_article=np.mean(trip_art) if trip_art else float("nan"),
                     reach_ref_article=np.mean(reach) if reach else float("nan"),
                     ref_numbers_in_ctx=np.mean(num_ctx) if num_ctx else float("nan"),
                     ref_numbers_in_article_subgraph=np.mean(num_graph) if num_graph else float("nan"),
                     idk=int(g["response"].fillna("").str.strip().str.lower().str.startswith(IDK).sum()),
                     ctx_chars=int(g["retrieved_contexts"].fillna("").str.len().mean()))
        out.append(stats)
    res = pd.DataFrame(out).round(3)
    print(res.to_string(index=False))
    res.to_csv(OUT / "graph_retrieval_metrics.csv", index=False)


# ----------------------------------------------------------------------------- report
def report():
    import pandas as pd, numpy as np
    frames = []
    v6 = pd.read_csv(TAGS / "v6" / "merged_final_v6.csv", encoding="utf-8-sig"); v6["ver"] = "v6"; v6["run"] = 1
    frames.append(v6)
    for tag in COMPARE_TAGS:
        for p in sorted((TAGS / tag).glob(f"merged_final_{tag}_run*.csv")):
            d = pd.read_csv(p, encoding="utf-8-sig"); d["ver"] = tag
            d["run"] = int(re.search(r"run(\d+)", p.name).group(1))
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["idk"] = df["response"].fillna("").str.strip().str.lower().str.startswith(IDK)
    cols = [c for c in METRIC_COLS + EXTRA_COLS if c in df.columns]
    rows = []
    for (ver, mode), d in df.groupby(["ver", "mode"]):
        row = {"ver": ver, "mode": mode, "runs": d["run"].nunique(), "n": len(d) // max(1, d["run"].nunique())}
        per_run = d.groupby("run")
        row["idk"] = f"{per_run['idk'].sum().mean():.1f}"
        for c in cols:
            m = per_run[c].mean()
            if len(m) > 1:
                ci = 1.96 * m.std(ddof=1) / math.sqrt(len(m))
                row[c] = f"{m.mean():.3f}±{ci:.3f}"
            else:
                row[c] = f"{m.mean():.3f}"
        rows.append(row)
    res = pd.DataFrame(rows).sort_values(["mode", "ver"])
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))
    res.to_csv(OUT / f"report_v6_vs_{'_'.join(COMPARE_TAGS)}.csv", index=False)
    # paired per-question comparison (graph mode, mean over runs) vs v6
    g6 = df[(df.ver == "v6") & (df["mode"] == "graph")].set_index("user_input")
    for tag in COMPARE_TAGS:
        gt = df[(df.ver == tag) & (df["mode"] == "graph")].groupby("user_input")[cols].mean()
        both = g6.join(gt, rsuffix="_t", how="inner")
        for c in ["factual_correctness(mode=f1)", "faithfulness", "answer_accuracy", "context_relevance"]:
            if c in both.columns and c + "_t" in both.columns:
                delta = (both[c + "_t"] - both[c]).dropna()
                wins, losses = (delta > 0.05).sum(), (delta < -0.05).sum()
                rng = np.random.default_rng(0)
                boots = [np.mean(rng.choice(delta.values, len(delta))) for _ in range(2000)]
                print(f"paired graph {tag}-v6 {c}: mean delta={delta.mean():+.3f} (95% CI {np.percentile(boots, 2.5):+.3f}..{np.percentile(boots, 97.5):+.3f}), wins={wins} losses={losses} n={len(delta)}")


if __name__ == "__main__":
    load_openai_key()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "score", "graphmetrics", "report"])
    ap.add_argument("--modes", nargs="+", default=["graph"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--run-ids", nargs="+", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--files", nargs="+", default=None)
    ap.add_argument("--compare", nargs="+", default=None, help="tags for the report (default: current TAG)")
    a = ap.parse_args()
    COMPARE_TAGS = a.compare or [TAG]
    globals()["COMPARE_TAGS"] = COMPARE_TAGS
    if a.stage == "collect":
        asyncio.run(collect(a.modes, a.runs, a.concurrency, a.limit))
    elif a.stage == "score":
        score(a.run_ids or list(range(1, a.runs + 1)), a.batch)
    elif a.stage == "graphmetrics":
        graphmetrics(a.files or [str(TAGS / "v6" / "merged_final_v6.csv")] + [str(p) for t in COMPARE_TAGS for p in sorted((TAGS / t).glob(f"qa_results_{t}_run*.csv"))])
    else:
        report()
