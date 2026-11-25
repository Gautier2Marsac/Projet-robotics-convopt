import pydot
import numpy as np

from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    Hyperellipsoid,
    Point,
)
from pydrake.solvers import (
    CommonSolverOption,
    GurobiSolver,
    MathematicalProgram,
    MosekSolver,
    SolverOptions,
)
from pydrake.all import le

from gcs.rounding import MipPathExtraction


def polytopeDimension(A, b, tol=1e-4):

    assert A.shape[0] == b.size

    m, n = A.shape
    eq = []

    while True:

        ineq = [i for i in range(m) if i not in eq]
        A_ineq = A[ineq]
        b_ineq = b[ineq]

        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(n)
        r = prog.NewContinuousVariables(1)[0]

        if len(eq) > 0:
            prog.AddLinearEqualityConstraint(A_eq.dot(x), b_eq)
        if len(ineq) > 0:
            c = prog.AddLinearConstraint(
                le(A_ineq.dot(x) + r * np.ones(len(ineq)), b_ineq)
            )
        prog.AddBoundingBoxConstraint(0, 1, r)

        prog.AddLinearCost(-r)

        solver = MosekSolver()
        result = solver.Solve(prog)

        if not result.is_success():
            return -1

        if result.GetSolution(r) > tol:
            eq_rank = 0 if len(eq) == 0 else np.linalg.matrix_rank(A_eq)
            return n - eq_rank

        c_opt = np.abs(result.GetDualSolution(c))
        eq += [ineq[i] for i, ci in enumerate(c_opt) if ci > tol]
        A_eq = A[eq]
        b_eq = b[eq]


class BaseGCS:
    def __init__(self, regions, num_layers=1):
        assert isinstance(num_layers, int) and num_layers >= 1
        self.names = None
        if type(regions) is dict:
            self.names = list(regions.keys())
            regions = list(regions.values())
        else:
            self.names = ["v" + str(ii) for ii in range(len(regions))]
        self.dimension = regions[0].ambient_dimension()
        self.regions = regions.copy()
        self.num_regions = len(self.regions)
        self.num_layers = num_layers
        self.rounding_fn = []
        self.rounding_kwargs = {}
        for r in self.regions:
            assert r.ambient_dimension() == self.dimension

        self.gcs = GraphOfConvexSets()
        self.options = GraphOfConvexSetsOptions()
        self.source = None
        self.target = None

    def vertexIndex(self, layer, region_idx):
        """Global index in self.gcs.Vertices() for (layer, region_idx)."""
        assert 0 <= layer < self.num_layers
        assert 0 <= region_idx < self.num_regions
        return layer * self.num_regions + region_idx

    def vertexName(self, layer, region_idx):
        """Readable vertex name (preserves original naming when num_layers == 1)."""
        base_name = self.names[region_idx] if self.names is not None else f"v{region_idx}"
        if self.num_layers == 1:
            return base_name
        return f"L{layer}_{base_name}"

    def addSourceTarget(self, source, target, edges=None):
        if self.source is not None:
            self.gcs.RemoveVertex(self.source)
            self.source = None
        if self.target is not None:
            self.gcs.RemoveVertex(self.target)
            self.target = None

        assert len(source) == self.dimension
        assert len(target) == self.dimension

        # Vertices are ordered layer by layer before adding source and target
        vertices = self.gcs.Vertices()
        self.source = self.gcs.AddVertex(Point(source), "source")
        self.target = self.gcs.AddVertex(Point(target), "target")

        # Add edges connecting source (to layer 0) and target (from last layer)
        if edges is None:
            edges = self.findStartGoalEdges(source, target)

        if not (len(edges[0]) > 0):
            raise ValueError("Source vertex is not connected.")
        if not (len(edges[1]) > 0):
            raise ValueError("Target vertex is not connected.")

        source_edges = []
        target_edges = []
        for ii in edges[0]:
            u = vertices[ii]
            edge = self.gcs.AddEdge(self.source, u, f"(source, {u.name()})")
            source_edges.append(edge)

        for ii in edges[1]:
            u = vertices[ii]
            edge = self.gcs.AddEdge(u, self.target, f"({u.name()}, target)")
            target_edges.append(edge)

        return source_edges, target_edges

    def findEdgesViaOverlaps(self):
        edges = []
        for ii in range(self.num_regions):
            for jj in range(ii + 1, self.num_regions):
                if self.regions[ii].IntersectsWith(self.regions[jj]):
                    edges.append((ii, jj))
                    edges.append((jj, ii))
        return edges

    def findEdgesViaFullDimensionOverlaps(self):
        edges = []
        for ii in range(self.num_regions):
            for jj in range(ii + 1, self.num_regions):
                A = np.vstack((self.regions[ii].A(), self.regions[jj].A()))
                b = np.concatenate((self.regions[ii].b(), self.regions[jj].b()))
                if polytopeDimension(A, b) >= self.dimension - 1:
                    edges.append((ii, jj))
                    edges.append((jj, ii))
        return edges

    def findLayeredIntraEdges(self, base_edges=None, full_dim_overlap=False):
        """Duplicate intra-layer region adjacency edges across all layers."""
        if base_edges is None:
            if full_dim_overlap:
                base_edges = self.findEdgesViaFullDimensionOverlaps()
            else:
                base_edges = self.findEdgesViaOverlaps()

        layered_edges = []
        for layer in range(self.num_layers):
            for ii, jj in base_edges:
                layered_edges.append(
                    (self.vertexIndex(layer, ii), self.vertexIndex(layer, jj))
                )
        return layered_edges

    def findWaypointRegions(self, waypoint, radius=0.0):
        """Return region indices containing waypoint (or intersecting ball of radius)."""
        waypoint = np.asarray(waypoint, dtype=float)
        assert waypoint.shape == (self.dimension,)
        assert radius >= 0.0

        if radius == 0.0:
            return [
                ii
                for ii in range(self.num_regions)
                if self.regions[ii].PointInSet(waypoint)
            ]

        ball = Hyperellipsoid.MakeHypersphere(radius, waypoint)
        return [
            ii
            for ii in range(self.num_regions)
            if self.regions[ii].IntersectsWith(ball)
        ]

    def findWaypointTransitionEdges(
        self, waypoints, radii=None, allow_cross_region=True, ordered=True
    ):
        """
        Compute inter-layer transition edges for a sequence/set of waypoints.

        - If ordered=True (num_layers == N + 1):
            connects layer k to layer k + 1 through regions containing waypoints[k].
        - If ordered=False (TSP Bitmask DAG, num_layers == 2**N):
            for every visited-subset bitmask `mask` and every unvisited waypoint k,
            connects layer `mask` to layer `mask | (1 << k)` through regions
            containing waypoints[k].

        Returns a list of tuples: (wp_idx, u_idx, v_idx, waypoint, radius).
        """
        num_wp = len(waypoints)
        expected_layers = (num_wp + 1) if ordered else (1 << num_wp)
        if self.num_layers != expected_layers:
            raise ValueError(
                f"Expected {expected_layers} layers for {num_wp} waypoints "
                f"(ordered={ordered}), got {self.num_layers}."
            )

        if radii is None:
            radii = [0.0] * num_wp
        elif isinstance(radii, (int, float)):
            radii = [float(radii)] * num_wp
        else:
            assert len(radii) == num_wp

        # Precompute safe regions and valid (ii, jj) region pairs for each waypoint
        wp_data = []
        for k, (wp, rad) in enumerate(zip(waypoints, radii)):
            wp_arr = np.asarray(wp, dtype=float)
            wp_regions = self.findWaypointRegions(wp_arr, radius=rad)
            if len(wp_regions) == 0:
                raise ValueError(
                    f"Waypoint {k} ({wp_arr}) with radius {rad} is not contained in any safe region."
                )
            region_pairs = [
                (ii, jj)
                for ii in wp_regions
                for jj in wp_regions
                if (allow_cross_region or ii == jj)
                and (ii == jj or self.regions[ii].IntersectsWith(self.regions[jj]))
            ]
            wp_data.append((wp_arr, float(rad), region_pairs))

        transition_edges = []
        if ordered:
            for k, (wp_arr, rad, region_pairs) in enumerate(wp_data):
                for ii, jj in region_pairs:
                    u_idx = self.vertexIndex(k, ii)
                    v_idx = self.vertexIndex(k + 1, jj)
                    transition_edges.append((k, u_idx, v_idx, wp_arr, rad))
        else:
            for mask in range(1 << num_wp):
                for k in range(num_wp):
                    if mask & (1 << k):
                        continue  # Waypoint k already visited in this state
                    next_mask = mask | (1 << k)
                    wp_arr, rad, region_pairs = wp_data[k]
                    for ii, jj in region_pairs:
                        u_idx = self.vertexIndex(mask, ii)
                        v_idx = self.vertexIndex(next_mask, jj)
                        transition_edges.append((k, u_idx, v_idx, wp_arr, rad))

        return transition_edges

    def findStartGoalEdges(self, start, goal):
        """Connect start to layer 0 and goal to the final layer (num_layers - 1)."""
        edges = [[], []]
        last_layer = self.num_layers - 1
        for ii in range(self.num_regions):
            if self.regions[ii].PointInSet(start):
                edges[0].append(self.vertexIndex(0, ii))
            if self.regions[ii].PointInSet(goal):
                edges[1].append(self.vertexIndex(last_layer, ii))
        return edges

    def setSolver(self, solver):
        self.options.solver = solver

    def setSolverOptions(self, options):
        self.options.solver_options = options

    def setPaperSolverOptions(self):
        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        solver_options.SetOption(
            MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3
        )
        solver_options.SetOption(
            MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1
        )
        solver_options.SetOption(
            MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3
        )
        solver_options.SetOption(
            MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", 3600.0
        )
        solver_options.SetOption(GurobiSolver.id(), "MIPGap", 1e-3)
        solver_options.SetOption(GurobiSolver.id(), "TimeLimit", 3600.0)
        self.options.solver_options = solver_options

    def setRoundingStrategy(self, rounding_fn, **kwargs):
        self.rounding_kwargs = kwargs
        if callable(rounding_fn):
            self.rounding_fn = [rounding_fn]
        elif isinstance(rounding_fn, list):
            assert len(rounding_fn) > 0
            for fn in rounding_fn:
                assert callable(fn)
            self.rounding_fn = rounding_fn
        else:
            raise ValueError(
                "Rounding strategy must either be a function or list of functions."
            )

    def ResetGraph(self, vertices=None):
        if vertices is None:
            vertices = [v for v in (self.source, self.target) if v is not None]
            self.source = None
            self.target = None
        for v in vertices:
            if v is not None:
                self.gcs.RemoveVertex(v)
        for edge in self.gcs.Edges():
            edge.ClearPhiConstraints()

    def VisualizeGraph(self, file_type="svg"):
        graphviz = self.gcs.GetGraphvizString(None, False)
        data = pydot.graph_from_dot_data(graphviz)[0]
        if file_type == "svg":
            return data.create_svg()
        elif file_type == "png":
            return data.create_png()
        else:
            raise ValueError("Unrecognized file type:", file_type)

    def solveGCS(self, rounding, preprocessing, verbose):

        results_dict = {}
        self.options.convex_relaxation = rounding
        self.options.preprocessing = preprocessing
        self.options.max_rounded_paths = 0

        result = self.gcs.SolveShortestPath(self.source, self.target, self.options)

        if rounding:
            results_dict["relaxation_result"] = result
            results_dict["relaxation_solver_time"] = (
                result.get_solver_details().optimizer_time
            )
            results_dict["relaxation_cost"] = result.get_optimal_cost()
        else:
            results_dict["mip_result"] = result
            results_dict["mip_solver_time"] = (
                result.get_solver_details().optimizer_time
            )
            results_dict["mip_cost"] = result.get_optimal_cost()

        if not result.is_success():
            print("First solve failed")
            return None, None, results_dict

        if verbose:
            print(
                "Solution\t",
                "Success:",
                result.get_solution_result(),
                "Cost:",
                result.get_optimal_cost(),
                "Solver time:",
                result.get_solver_details().optimizer_time,
            )

        # Solve with hard edge choices
        if rounding and len(self.rounding_fn) > 0:
            # Extract path
            active_edges = []
            found_path = False
            for fn in self.rounding_fn:
                rounded_edges = fn(
                    self.gcs,
                    result,
                    self.source,
                    self.target,
                    **self.rounding_kwargs,
                )
                if rounded_edges is None:
                    print(fn.__name__, "could not find a path.")
                    active_edges.append(rounded_edges)
                else:
                    found_path = True
                    active_edges.extend(rounded_edges)
            results_dict["rounded_paths"] = active_edges
            if not found_path:
                print("All rounding strategies failed to find a path.")
                return None, None, results_dict

            self.options.preprocessing = False
            rounded_results = []
            best_cost = np.inf
            best_path = None
            best_result = None
            max_rounded_solver_time = 0.0
            total_rounded_solver_time = 0.0
            for path_edges in active_edges:
                if path_edges is None:
                    rounded_results.append(None)
                    continue
                for edge in self.gcs.Edges():
                    if edge in path_edges:
                        edge.AddPhiConstraint(True)
                    else:
                        edge.AddPhiConstraint(False)
                rounded_results.append(
                    self.gcs.SolveShortestPath(
                        self.source, self.target, self.options
                    )
                )
                solve_time = (
                    rounded_results[-1].get_solver_details().optimizer_time
                )
                max_rounded_solver_time = np.maximum(
                    solve_time, max_rounded_solver_time
                )
                total_rounded_solver_time += solve_time
                if (
                    rounded_results[-1].is_success()
                    and rounded_results[-1].get_optimal_cost() < best_cost
                ):
                    best_cost = rounded_results[-1].get_optimal_cost()
                    best_path = path_edges
                    best_result = rounded_results[-1]

            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["rounded_results"] = rounded_results
            results_dict["max_rounded_solver_time"] = max_rounded_solver_time
            results_dict["total_rounded_solver_time"] = (
                total_rounded_solver_time
            )
            if best_result is not None:
                results_dict["rounded_cost"] = best_result.get_optimal_cost()
                if results_dict["relaxation_cost"] > 0:
                    results_dict["certified_gap"] = (
                        results_dict["rounded_cost"]
                        - results_dict["relaxation_cost"]
                    ) / results_dict["relaxation_cost"]

            if verbose:
                print("Rounded Solutions:")
                for r in rounded_results:
                    if r is None:
                        print("\t\tNo path to solve")
                        continue
                    print(
                        "\t\t",
                        "Success:",
                        r.get_solution_result(),
                        "Cost:",
                        r.get_optimal_cost(),
                        "Solver time:",
                        r.get_solver_details().optimizer_time,
                    )

            if best_path is None:
                print("Second solve failed on all paths.")
                return best_path, best_result, results_dict
        elif rounding:
            self.options.max_rounded_paths = 10

            rounded_result = self.gcs.SolveShortestPath(
                self.source, self.target, self.options
            )
            if not rounded_result.is_success():
                print("Native Drake rounding failed.")
                return None, None, results_dict
            best_path = MipPathExtraction(
                self.gcs, rounded_result, self.source, self.target
            )[0]
            best_result = rounded_result
            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["rounded_results"] = [rounded_result]
            results_dict["rounded_cost"] = best_result.get_optimal_cost()
            if results_dict["relaxation_cost"] > 0:
                results_dict["certified_gap"] = (
                    results_dict["rounded_cost"]
                    - results_dict["relaxation_cost"]
                ) / results_dict["relaxation_cost"]
        else:
            best_path = MipPathExtraction(
                self.gcs, result, self.source, self.target
            )[0]
            best_result = result
            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["mip_path"] = best_path

        if verbose:
            for edge in best_path:
                print("Added", edge.name(), "to path.")

        return best_path, best_result, results_dict