"""
Sharded ErG v2 index + exact search.

  python erg_index.py build  --csv big.csv --smiles-col Charged_Smiles --id-col ID --out idx_dir [--shard-rows 4000000] [--limit N] [-j 16]
  python erg_index.py search --index idx_dir --smiles "CCO..." [--smiles-file q.smi] [--k 50] [--metric cosine|tanimoto] [--out hits.csv]
  python erg_index.py info   --index idx_dir

Index layout (idx_dir/):
  meta.json                   version, ncol, shard list (name, n)
  shard_00000.raw.bin         uint8 [n, 321] row-major, raw (un-fuzzed) ErG v2 histogram
  shard_00000.sqnorm.npy      float32 [n]  ||fuzzy(d)||^2  (fuzzy smoothing never stored)
  shard_00000.ids.parquet     id, smiles   (row groups of 65536 -> cheap lookup of hits)
Only the QUERY is smoothed at search time (<Fq,Fd> = <F^2 q, d>), so the database stays 321 B/molecule.
Similarity is in [0,1]; distance = 1 - similarity.  Molecules that fail to parse are not indexed.
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import erg_v2 as E

NCOL = len(E.COLUMNS)
ROW_GROUP = 65536


# ------------------------------------------------------------------ build ------------
def build(csv, out, smiles_col, id_col, shard_rows=4_000_000, limit=None, jobs=0, log=print):
    import pandas as pd
    import pyarrow as pa, pyarrow.parquet as pq
    os.makedirs(out, exist_ok=True)
    meta_path = os.path.join(out, "meta.json")
    meta = {"version": "erg_v2", "ncol": NCOL, "columns_hint": "6 counts + 21 pair types x 15 bins",
            "shards": [], "source_csv": os.path.abspath(csv)}
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        meta["shards"] = old["shards"]
    done = {s["name"]: s["n"] for s in meta["shards"]}
    reader = pd.read_csv(csv, usecols=[smiles_col, id_col], chunksize=shard_rows)
    t0, rows_seen, total_fail = time.time(), 0, 0
    for k, df in enumerate(reader):
        if limit is not None and rows_seen >= limit:
            break
        if limit is not None:
            df = df.iloc[: limit - rows_seen]
        rows_seen += len(df)
        name = f"shard_{k:05d}"
        if name in done:
            log(f"[{name}] already built ({done[name]} mols) - skipped")
            continue
        t1 = time.time()
        raw, ok = E.raw_many(df[smiles_col].astype(str).tolist(), jobs=jobs, progress=False)
        fail = int((~ok).sum()); total_fail += fail
        raw = np.ascontiguousarray(raw[ok])
        sq = E.fuzzy_sqnorm(raw)
        sub = df[ok].reset_index(drop=True)
        tmp = lambda ext: os.path.join(out, f"{name}.{ext}.tmp")
        fin = lambda ext: os.path.join(out, f"{name}.{ext}")
        raw.tofile(tmp("raw.bin"))
        with open(tmp("sqnorm.npy"), "wb") as fh:
            np.save(fh, sq)
        pq.write_table(pa.table({"id": sub[id_col].to_numpy(), "smiles": sub[smiles_col].astype(str).to_numpy()}),
                       tmp("ids.parquet"), row_group_size=ROW_GROUP, compression="zstd")
        os.replace(tmp("raw.bin"), fin("raw.bin"))
        os.replace(tmp("sqnorm.npy"), fin("sqnorm.npy"))
        os.replace(tmp("ids.parquet"), fin("ids.parquet"))   # commit marker
        meta["shards"].append({"name": name, "n": int(len(raw))})
        json.dump(meta, open(meta_path, "w"), indent=1)
        dt = time.time() - t1
        log(f"[{name}] {len(raw)} mols ({fail} failed) in {dt:.0f}s = {len(df)/dt:.0f} mol/s; "
            f"total rows {rows_seen}, elapsed {(time.time()-t0)/60:.1f} min")
    meta["n_total"] = int(sum(s["n"] for s in meta["shards"]))
    json.dump(meta, open(meta_path, "w"), indent=1)
    log(f"done: {meta['n_total']} molecules in {len(meta['shards'])} shards ({total_fail} failed this run)")
    return meta


# ------------------------------------------------------------------ search -----------
class Index:
    def __init__(self, path):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        self.shards = self.meta["shards"]
        self.offsets = np.cumsum([0] + [s["n"] for s in self.shards])
        self.n = int(self.offsets[-1])
        self._sq = {}

    def _f(self, name, ext):
        return os.path.join(self.path, f"{name}.{ext}")

    def sqnorm(self, si):
        if si not in self._sq:
            self._sq[si] = np.load(self._f(self.shards[si]["name"], "sqnorm.npy"))
        return self._sq[si]

    def _tasks(self, chunk):
        for si, s in enumerate(self.shards):
            for a in range(0, s["n"], chunk):
                yield si, a, min(chunk, s["n"] - a)

    def search(self, query_smiles, k=50, metric="cosine", chunk=262144, workers=8, log=None):
        """Exact top-k for each query SMILES. Returns list (per query) of dicts with global_row, similarity."""
        Q = len(query_smiles)
        qraw, qok = E.raw_many(list(query_smiles), jobs=1, progress=False, backend="python")
        if not qok.all():
            raise ValueError(f"unparsable query SMILES at positions {np.where(~qok)[0].tolist()}")
        q2 = E._smooth(qraw, E.F2).T.copy()                      # [321, Q]  F^2 q
        qf = E.fuzzy_expand(qraw)
        qn2 = (qf * qf).sum(1).astype(np.float32)                # [Q]
        best_s = [np.empty(0, np.float32) for _ in range(Q)]
        best_i = [np.empty(0, np.int64) for _ in range(Q)]
        thresh = np.full(Q, -1.0, np.float32)

        tl = __import__("threading").local()

        def work(task):
            si, a, n = task
            buf = getattr(tl, "buf", None)
            if buf is None or buf.size < chunk * NCOL:
                buf = tl.buf = np.empty(chunk * NCOL, np.uint8)
            with open(self._f(self.shards[si]["name"], "raw.bin"), "rb", buffering=0) as f:
                f.seek(a * NCOL)
                mv = memoryview(buf)[: n * NCOL]
                got = 0
                while got < n * NCOL:
                    r = f.readinto(mv[got:])
                    if not r: raise IOError("short read")
                    got += r
            blk = buf[: n * NCOL].reshape(n, NCOL).astype(np.float32)
            dot = blk @ q2                                        # [n, Q]
            n2 = self.sqnorm(si)[a:a + n][:, None]
            if metric == "dot":
                sim = dot
            else:
                den = np.sqrt(n2 * qn2[None, :]) if metric == "cosine" else (n2 + qn2[None, :] - dot)
                with np.errstate(divide="ignore", invalid="ignore"):
                    sim = np.where(den > 0, dot / den, 0.0)
                np.clip(sim, 0.0, 1.0, out=sim)
            out = []
            base = int(self.offsets[si]) + a
            for qi in range(Q):
                col = sim[:, qi]
                kk = min(k, n)
                if kk < n:
                    idx = np.argpartition(-col, kk - 1)[:kk]
                else:
                    idx = np.arange(n)
                out.append((idx + base, col[idx].astype(np.float32)))
            return out

        t0 = time.time(); done = 0
        tasks = list(self._tasks(chunk))
        with ThreadPoolExecutor(workers) as ex:
            for res in ex.map(work, tasks):
                for qi, (gi, sc) in enumerate(res):
                    bi, bs = np.concatenate([best_i[qi], gi]), np.concatenate([best_s[qi], sc])
                    if len(bs) > 4 * k:
                        keep = np.argpartition(-bs, k - 1)[:k]
                        bi, bs = bi[keep], bs[keep]
                    best_i[qi], best_s[qi] = bi, bs
                done += 1
                if log and done % 50 == 0:
                    log(f"  scanned {done}/{len(tasks)} chunks, {time.time()-t0:.0f}s")
        results = []
        for qi in range(Q):
            o = np.argsort(-best_s[qi])[:k]
            results.append({"global_row": best_i[qi][o], "similarity": best_s[qi][o]})
        return results

    def fetch(self, global_rows):
        """(id, smiles) for global row numbers, reading only the needed parquet row groups."""
        import pyarrow.parquet as pq
        global_rows = np.asarray(global_rows, dtype=np.int64)
        ids = np.empty(len(global_rows), dtype=object); smi = np.empty(len(global_rows), dtype=object)
        si = np.searchsorted(self.offsets, global_rows, side="right") - 1
        for s in np.unique(si):
            sel = np.where(si == s)[0]
            local = global_rows[sel] - self.offsets[s]
            pf = pq.ParquetFile(self._f(self.shards[s]["name"], "ids.parquet"))
            groups = np.unique(local // ROW_GROUP)
            tab = pf.read_row_groups(groups.tolist()).to_pydict()
            # position of each group's first row inside the concatenated table
            pos = {}
            cur = 0
            for g in groups:
                n = pf.metadata.row_group(int(g)).num_rows
                pos[int(g)] = cur; cur += n
            for j, l in zip(sel, local):
                g = int(l // ROW_GROUP)
                r = pos[g] + int(l - g * ROW_GROUP)
                ids[j], smi[j] = tab["id"][r], tab["smiles"][r]
        return ids, smi


# ------------------------------------------------------------------ CLI --------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--csv", required=True); b.add_argument("--out", required=True)
    b.add_argument("--smiles-col", default="SMILES"); b.add_argument("--id-col", default="ID")
    b.add_argument("--shard-rows", type=int, default=4_000_000); b.add_argument("--limit", type=int)
    b.add_argument("-j", "--jobs", type=int, default=0, help="threads for the C++ kernel (0 = all)")
    s = sp.add_parser("search"); s.add_argument("--index", required=True)
    s.add_argument("--smiles", action="append", default=[]); s.add_argument("--smiles-file")
    s.add_argument("--k", type=int, default=50); s.add_argument("--metric", default="cosine", choices=("cosine", "tanimoto", "dot"))
    s.add_argument("--workers", type=int, default=8); s.add_argument("--out")
    i = sp.add_parser("info"); i.add_argument("--index", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.csv, a.out, a.smiles_col, a.id_col, a.shard_rows, a.limit, a.jobs)
    elif a.cmd == "info":
        ix = Index(a.index); print(f"{ix.n} molecules, {len(ix.shards)} shards, {ix.n*NCOL/1e9:.2f} GB raw")
    else:
        import pandas as pd
        qs = list(a.smiles)
        if a.smiles_file:
            qs += [l.split()[0] for l in open(a.smiles_file) if l.strip()]
        ix = Index(a.index); t = time.time()
        res = ix.search(qs, k=a.k, metric=a.metric, workers=a.workers, log=lambda m: print(m, file=sys.stderr))
        rows = []
        for q, r in zip(qs, res):
            ids, smi = ix.fetch(r["global_row"])
            for rank, (i_, s_, sim) in enumerate(zip(ids, smi, r["similarity"]), 1):
                rows.append({"query": q, "rank": rank, "id": i_, "similarity": float(sim), "distance": 1 - float(sim), "smiles": s_})
        df = pd.DataFrame(rows)
        print(f"searched {ix.n} molecules x {len(qs)} queries in {time.time()-t:.1f}s", file=sys.stderr)
        if a.out: df.to_csv(a.out, index=False)
        else: print(df.to_string(index=False, max_colwidth=70))


if __name__ == "__main__":
    main()
