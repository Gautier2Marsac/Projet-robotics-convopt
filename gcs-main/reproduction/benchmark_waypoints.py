"""
Benchmark of our waypoint extensions on random instances.

Unordered waypoints (section 5 of the poster), three methods per instance:
  - enum      : solve the ordered fly-by GCS for every permutation (N! solves) -> reference optimum
  - heuristic : best order for straight-line distances (open TSP), then ONE ordered GCS
  - heldkarp  : our Held-Karp layered graph (2^N layers), ONE GCS solve
  - mip       : (optional, --mip) Held-Karp graph solved exactly (no relaxation)

Fly-by waypoints (section 4), optional --flyby, per instance with 2 waypoints:
  - seq_c1    : chained problems, stop (v = 0) at each waypoint
  - seq_c2    : chained problems, v = a = 0 at each waypoint
  - flyby     : fly-by GCS, exact passage
  - flyby_eps : fly-by GCS, tolerance ball eps

Results are appended row by row to a CSV (re-running the same command resumes).

Usage (from gcs-main/):
  python reproduction/benchmark_waypoints.py --n_inst 10 --Ns 2 3 4 --flyby
  python reproduction/benchmark_waypoints.py --plot          # figures + summary from the CSV
"""
import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GCS_ROOT = os.environ.get("GCS_ROOT", os.path.dirname(HERE))
sys.path.insert(0, GCS_ROOT)

FIELDS = ["bench", "N", "inst", "method", "ok", "wall_s", "J", "T",
          "cert_gap", "order", "waypoints"]


# setup
def load_env():
    from scipy.spatial import ConvexHull
    from pydrake.geometry.optimization import HPolyhedron
    from models.env_2d import vertices

    def make_hpolytope(V):
        ch = ConvexHull(V)
        return HPolyhedron(ch.equations[:, :-1], -ch.equations[:, -1])

    regions = [make_hpolytope(V) for V in vertices]
    x_min = np.min(np.vstack(vertices), axis=0)
    x_max = np.max(np.vstack(vertices), axis=0)
    return regions, x_min, x_max


X_START = np.array([0.2, 0.2])
X_GOAL = np.array([4.8, 4.8])
ORDER, CONTINUITY = 6, 2
REGULARIZER = [1e-1, 1e-1]
HDOT_MIN = 1e-1
QDOT = 1.0


def make_solver(name):
    from pydrake.solvers import ClarabelSolver, MosekSolver
    if name == "mosek":
        s = MosekSolver()
        if not s.enabled():
            sys.exit("MOSEK is not enabled (license not found). Use --solver clarabel to test.")
        return s
    return ClarabelSolver()


def configure(g, solver_name):
    from pydrake.solvers import CommonSolverOption
    g.setSolver(make_solver(solver_name))
    if solver_name == "mosek":
        g.setPaperSolverOptions()
        g.options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 0)


def sample_waypoints(rng, regions, x_min, x_max, n, min_dist=0.6, margin=0.15):
    """Uniform samples in free space, away from start/goal and from each other."""
    pts = []
    tries = 0
    while len(pts) < n:
        tries += 1
        if tries > 100000:
            raise RuntimeError("could not sample waypoints")
        p = rng.uniform(x_min + margin, x_max - margin)
        if not any(r.PointInSet(p) for r in regions):
            continue
        others = pts + [X_START, X_GOAL]
        if min(np.linalg.norm(p - q) for q in others) < min_dist:
            continue
        pts.append(p)
    return pts


#  solvers
def solve_bezier(regions, solver_name, wps=None, ordered=True, eps=0.0,
                 src=X_START, tgt=X_GOAL, velocity="zero", zero_deriv=None):
    from gcs.bezier import BezierGCS
    g = BezierGCS(regions, ORDER, CONTINUITY, hdot_min=HDOT_MIN,
                  waypoints=wps, waypoint_radii=eps, ordered=ordered)
    g.addTimeCost(1)
    g.addVelocityLimits([-QDOT] * 2, [QDOT] * 2)
    g.addDerivativeRegularization(*REGULARIZER, 2)
    if zero_deriv is not None:
        g.addSourceTarget(src, tgt, zero_deriv_boundary=zero_deriv)
    else:
        g.addSourceTarget(src, tgt, velocity=np.zeros((2, 2)))
    configure(g, solver_name)
    return g


def run(g, rounding=True):
    t0 = time.perf_counter()
    try:
        traj, res = g.SolvePath(rounding=rounding)
    except Exception as e:  # noqa: BLE001
        print("   solve failed:", e)
        traj, res = None, {}
    wall = time.perf_counter() - t0
    if traj is None:
        return dict(ok=0, wall_s=wall, J=np.nan, T=np.nan, cert_gap=np.nan, order="")
    J = res.get("rounded_cost", res.get("mip_cost", np.nan))
    gap = res.get("certified_gap", np.nan)
    order = res.get("visit_order", "")
    return dict(ok=1, wall_s=wall, J=float(J), T=float(traj.end_time()),
                cert_gap=float(gap) if gap is not None else np.nan,
                order="-".join(map(str, order)) if len(order) else "")


def heuristic_order(wps):
    """Open TSP on straight-line distances: start -> all waypoints -> goal."""
    best, best_perm = np.inf, None
    for perm in itertools.permutations(range(len(wps))):
        pts = [X_START] + [wps[i] for i in perm] + [X_GOAL]
        d = sum(np.linalg.norm(a - b) for a, b in zip(pts[:-1], pts[1:]))
        if d < best:
            best, best_perm = d, perm
    return list(best_perm)


#  csv
def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                done.add((row["bench"], int(row["N"]), int(row["inst"]), row["method"]))
    return done


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def fmt_wps(wps):
    return ";".join(f"{p[0]:.3f},{p[1]:.3f}" for p in wps)


#  benches
def bench_unordered(args, regions, x_min, x_max, done):
    for N in args.Ns:
        rng = np.random.default_rng(args.seed + 1000 * N)
        for inst in range(args.n_inst):
            wps = sample_waypoints(rng, regions, x_min, x_max, N)
            base = dict(bench="unordered", N=N, inst=inst, waypoints=fmt_wps(wps))
            print(f"[unordered] N={N} inst={inst}")

            def record(method, out):
                append(args.csv, {**base, "method": method, **out})
                print(f"   {method:10s} J={out['J']:.2f} wall={out['wall_s']:.1f}s gap={out['cert_gap']}")

            only_mip = args.mip and args.no_unordered

            # enumeration (reference optimum)
            if not only_mip and N <= args.enum_max and ("unordered", N, inst, "enum") not in done:
                t0 = time.perf_counter()
                best = dict(ok=0, J=np.inf, T=np.nan, cert_gap=np.nan, order="")
                for perm in itertools.permutations(range(N)):
                    out = run(solve_bezier(regions, args.solver, [wps[i] for i in perm]))
                    if out["ok"] and out["J"] < best["J"]:
                        best = {**out, "order": "-".join(map(str, perm))}
                best["wall_s"] = time.perf_counter() - t0
                if not best["ok"]:
                    best["J"] = np.nan
                record("enum", best)

            # heuristic: straight-line TSP order, then one ordered GCS
            if not only_mip and ("unordered", N, inst, "heuristic") not in done:
                t0 = time.perf_counter()
                perm = heuristic_order(wps)
                out = run(solve_bezier(regions, args.solver, [wps[i] for i in perm]))
                out["wall_s"] = time.perf_counter() - t0
                out["order"] = "-".join(map(str, perm))
                record("heuristic", out)

            # our Held-Karp layered graph, one relaxed solve + rounding
            if not only_mip and ("unordered", N, inst, "heldkarp") not in done:
                record("heldkarp", run(solve_bezier(regions, args.solver, wps, ordered=False)))

            # exact MIP on the Held-Karp graph (slow)
            if args.mip and N <= args.mip_max and ("unordered", N, inst, "mip") not in done:
                record("mip", run(solve_bezier(regions, args.solver, wps, ordered=False), rounding=False))


def bench_flyby(args, regions, x_min, x_max, done):
    rng = np.random.default_rng(args.seed + 777)
    for inst in range(args.n_inst):
        wps = sample_waypoints(rng, regions, x_min, x_max, 2)
        base = dict(bench="flyby", N=2, inst=inst, waypoints=fmt_wps(wps))
        print(f"[flyby] inst={inst}")

        def record(method, out):
            append(args.csv, {**base, "method": method, **out})
            print(f"   {method:10s} J={out['J']:.2f} T={out['T']:.2f} wall={out['wall_s']:.1f}s")

        pts = [X_START] + wps + [X_GOAL]
        for method, zd in [("seq_c1", None), ("seq_c2", 2)]:
            if ("flyby", 2, inst, method) in done:
                continue
            tot = dict(ok=1, wall_s=0.0, J=0.0, T=0.0, cert_gap=np.nan, order="")
            for a, b in zip(pts[:-1], pts[1:]):
                out = run(solve_bezier(regions, args.solver, src=a, tgt=b, zero_deriv=zd))
                if not out["ok"]:
                    tot["ok"] = 0
                for k in ("wall_s", "J", "T"):
                    tot[k] += out[k]
            record(method, tot)
        if ("flyby", 2, inst, "flyby") not in done:
            record("flyby", run(solve_bezier(regions, args.solver, wps, eps=0.0)))
        if ("flyby", 2, inst, "flyby_eps") not in done:
            record("flyby_eps", run(solve_bezier(regions, args.solver, wps, eps=args.eps)))


#  plots
def plot(args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NAVY, ORANGE, RED, GREY = "#1F3A5F", "#E07A1F", "#C0392B", "#7F8C8D"
    plt.rcParams.update({"font.size": 14, "savefig.dpi": 300, "savefig.bbox": "tight"})
    df = pd.read_csv(args.csv)
    out_dir = os.path.dirname(os.path.abspath(args.csv))

    u = df[(df.bench == "unordered") & (df.ok == 1)].copy()
    if len(u):
        # reference = enumeration if available, else best of all methods on that instance
        ref = u.groupby(["N", "inst"]).J.min().rename("J_ref")
        enum = u[u.method == "enum"].set_index(["N", "inst"]).J.rename("J_enum")
        u = u.join(ref, on=["N", "inst"]).join(enum, on=["N", "inst"])
        u["J_best"] = u.J_enum.fillna(u.J_ref)
        u["gap_pct"] = 100 * (u.J - u.J_best) / u.J_best
        u["found_opt"] = u.gap_pct < 0.1

        summ = u.groupby(["N", "method"]).agg(
            n=("J", "size"), wall_mean=("wall_s", "mean"), wall_std=("wall_s", "std"),
            gap_mean=("gap_pct", "mean"), gap_max=("gap_pct", "max"),
            opt_rate=("found_opt", "mean"), cert_gap_mean=("cert_gap", "mean"))
        summ["opt_rate"] *= 100
        summ["cert_gap_mean"] *= 100
        print("\n=== Unordered waypoints (gap vs best known order, %) ===")
        print(summ.round(2).to_string())
        summ.round(3).to_csv(os.path.join(out_dir, "summary_unordered.csv"))

        styles = {"enum": (RED, "Enumeration ($N!$ solves)"),
                  "heuristic": (GREY, "Straight-line TSP + 1 GCS"),
                  "heldkarp": (NAVY, "Held-Karp GCS (1 solve)"),
                  "mip": (ORANGE, "Held-Karp exact MIP")}
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        for m, (c, lab) in styles.items():
            s = summ.xs(m, level="method") if m in summ.index.get_level_values(1) else None
            if s is None:
                continue
            axes[0].errorbar(s.index, s.wall_mean, yerr=s.wall_std, color=c, marker="o", capsize=4, lw=2, label=lab)
            axes[1].plot(s.index, s.gap_mean, color=c, marker="o", lw=2, label=lab)
            axes[2].plot(s.index, s.opt_rate, color=c, marker="o", lw=2, label=lab)
        axes[0].set_yscale("log"); axes[0].set_ylabel("wall-clock time (s)"); axes[0].set_title("Runtime (mean ± std)")
        axes[1].set_ylabel("mean cost gap to best order (%)"); axes[1].set_title("Quality")
        axes[2].set_ylabel("instances with best order (%)"); axes[2].set_title("Optimal order found"); axes[2].set_ylim(-5, 105)
        for ax in axes:
            ax.set_xlabel("number of waypoints N"); ax.grid(alpha=0.3); ax.set_xticks(sorted(u.N.unique()))
        axes[0].legend(fontsize=11)
        plt.tight_layout()
        p = os.path.join(out_dir, "fig_unordered_benchmark.png")
        plt.savefig(p); print("saved", p)

    f = df[(df.bench == "flyby") & (df.ok == 1)].copy()
    if len(f):
        piv_T = f.pivot_table(index="inst", columns="method", values="T")
        piv_J = f.pivot_table(index="inst", columns="method", values="J")
        print("\n=== Fly-by (mean ± std over instances) ===")
        for m in ["seq_c1", "seq_c2", "flyby", "flyby_eps"]:
            if m in piv_T:
                print(f"{m:10s} T = {piv_T[m].mean():6.2f} ± {piv_T[m].std():5.2f}   "
                      f"J = {piv_J[m].mean():6.2f} ± {piv_J[m].std():5.2f}")
        if {"seq_c2", "flyby"} <= set(piv_T):
            dT = 100 * (piv_T.flyby - piv_T.seq_c2) / piv_T.seq_c2
            dJ = 100 * (piv_J.flyby - piv_J.seq_c2) / piv_J.seq_c2
            print(f"fly-by vs seq_c2: time {dT.mean():+.1f} % ± {dT.std():.1f}, cost {dJ.mean():+.1f} % ± {dJ.std():.1f}"
                  f"  (fly-by faster on {100 * (dT < 0).mean():.0f} % of instances)")
        cols = [m for m in ["seq_c1", "seq_c2", "flyby", "flyby_eps"] if m in piv_T]
        labels = {"seq_c1": "Seq. stop (C¹)", "seq_c2": "Seq. strict (C²)", "flyby": "Fly-by exact", "flyby_eps": f"Fly-by ε"}
        colors = {"seq_c1": "#E9A6A0", "seq_c2": RED, "flyby": NAVY, "flyby_eps": ORANGE}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, piv, lab in [(axes[0], piv_T, "trajectory duration T (s)"), (axes[1], piv_J, "total cost J")]:
            bp = ax.boxplot([piv[c].dropna() for c in cols], patch_artist=True, widths=0.6)
            for patch, c in zip(bp["boxes"], cols):
                patch.set_facecolor(colors[c])
            ax.set_xticklabels([labels[c] for c in cols], rotation=15)
            ax.set_ylabel(lab); ax.grid(axis="y", alpha=0.3)
        axes[0].set_title(f"Fly-by benchmark ({len(piv_T)} random instances)")
        plt.tight_layout()
        p = os.path.join(out_dir, "fig_flyby_benchmark.png")
        plt.savefig(p); print("saved", p)


#  main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="mosek", choices=["mosek", "clarabel"])
    ap.add_argument("--n_inst", type=int, default=10)
    ap.add_argument("--Ns", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--enum_max", type=int, default=4, help="largest N for brute-force enumeration")
    ap.add_argument("--mip", action="store_true", help="also solve the exact MIP (slow)")
    ap.add_argument("--mip_max", type=int, default=4)
    ap.add_argument("--flyby", action="store_true", help="also run the fly-by benchmark")
    ap.add_argument("--no_unordered", action="store_true",
                    help="skip the unordered benchmark (with --mip: run only the MIP rows)")
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=os.path.join(HERE, "benchmark_results", "results.csv"))
    ap.add_argument("--plot", action="store_true", help="only make figures/summary from the CSV")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)

    if args.plot:
        plot(args)
        return

    regions, x_min, x_max = load_env()
    done = load_done(args.csv)
    if args.flyby:
        bench_flyby(args, regions, x_min, x_max, done)
    if not args.no_unordered or args.mip:
        bench_unordered(args, regions, x_min, x_max, done)
    plot(args)


if __name__ == "__main__":
    main()
