import time
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters (given in the task)
# -----------------------------
k1 = 0.08
k2 = 0.15
k3 = 0.6
alpha = 0.05
Ta = 60.0
Meq = 0.12

# -----------------------------
# Global numeric settings
# -----------------------------
TOL = 1e-10
MAX_ITER = 100


# ============================================================
# 1) ODE system
# ============================================================
def ode_system(y, t):
    """
    Returns dy/dt = [dM/dt, dT/dt] as numpy array
    y = [M, T]
    """
    M, T = float(y[0]), float(y[1])
    E = np.exp(alpha * (T - Ta))
    dM = -k1 * (M - Meq) * E
    dT = k2 * (Ta - T) - k3 * dM
    return np.array([dM, dT], dtype=float)


def residual_implicit_euler(y_new, y_old, t_new, h):
    """
    Residual for implicit Euler:
    F(y_new) = y_new - y_old - h f(y_new, t_new)
    """
    return y_new - y_old - h * ode_system(y_new, t_new)


# ============================================================
# 2) Jacobian for implicit Euler residual (course required form)
# ============================================================
def compute_jacobian(y, h):
    """
    Full 2x2 Jacobian matrix for implicit Euler residual F = y - y_old - h f(y)

    Course given Jacobian entries (use these exact formulas):

    Let E = exp(alpha (T - Ta))

    J11 = 1 + h*k1*E
    J12 = h*k1*alpha*(M - Meq)*E
    J21 = -h*k3*k1*E
    J22 = 1 + h*k2 + h*k3*k1*alpha*(M - Meq)*E
    """
    M, T = float(y[0]), float(y[1])
    E = np.exp(alpha * (T - Ta))

    J11 = 1.0 + h * k1 * E
    J12 = h * k1 * alpha * (M - Meq) * E
    J21 = -h * k3 * k1 * E
    J22 = 1.0 + h * k2 + h * k3 * k1 * alpha * (M - Meq) * E

    return np.array([[J11, J12],
                     [J21, J22]], dtype=float)


# ============================================================
# 3) Gauss Seidel solver for linear system J x = b
# ============================================================
def gauss_seidel_solve(J, b, tol=1e-12, max_iter=50, x0=None):
    """
    Solve J x = b using Gauss Seidel iterations for a 2x2 system.

    Returns:
        x, gs_iter_count, history
    history stores ||x_k - x_{k-1}||_inf
    """
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = np.array(x0, dtype=float)

    hist = []

    # Safety checks
    if abs(J[0, 0]) < 1e-15 or abs(J[1, 1]) < 1e-15:
        # Diagonal too small, GS is unsafe
        return x, 0, np.array(hist, dtype=float)

    for it in range(max_iter):
        x_old = x.copy()

        # Update x[0] using current x[1]
        x[0] = (b[0] - J[0, 1] * x[1]) / J[0, 0]

        # Update x[1] using new x[0]
        x[1] = (b[1] - J[1, 0] * x[0]) / J[1, 1]

        diff = float(np.linalg.norm(x - x_old, ord=np.inf))
        hist.append(diff)

        if diff < tol:
            return x, it + 1, np.array(hist, dtype=float)

    return x, max_iter, np.array(hist, dtype=float)


# ============================================================
# 4) Implicit Euler solvers
# ============================================================
def implicit_euler_fixed_point(tspan, y0, h, tol=TOL, max_iter=80, store_steps=None):
    """
    Implicit Euler with fixed point iteration:
    y_{k+1} = y_old + h f(y_k, t_new)

    Returns:
        t, Y, iters_per_step, runtime, residual_histories
    residual_histories stores ||F|| for selected step indices
    """
    t0, t1 = float(tspan[0]), float(tspan[1])
    t = np.arange(t0, t1 + h, h)
    n_steps = len(t) - 1

    Y = np.zeros((len(t), 2), dtype=float)
    Y[0] = np.array(y0, dtype=float)

    iters = np.zeros(n_steps, dtype=int)
    residual_histories = {}

    start = time.time()

    for n in range(n_steps):
        y_old = Y[n].copy()
        t_new = t[n + 1]

        yk = y_old.copy()
        hist = []

        for k in range(max_iter):
            y_next = y_old + h * ode_system(yk, t_new)
            F = residual_implicit_euler(y_next, y_old, t_new, h)
            rnorm = float(np.linalg.norm(F, ord=2))
            hist.append(rnorm)

            if np.linalg.norm(y_next - yk, ord=np.inf) < tol:
                yk = y_next
                iters[n] = k + 1
                break

            yk = y_next
        else:
            iters[n] = max_iter

        Y[n + 1] = yk

        if store_steps is not None and n in store_steps:
            residual_histories[n] = np.array(hist, dtype=float)

    runtime = time.time() - start
    return t, Y, iters, runtime, residual_histories


def implicit_euler_newton_gauss_seidel(tspan, y0, h,
                                       newton_tol=TOL, newton_max_iter=30,
                                       gs_tol=1e-12, gs_max_iter=20,
                                       store_steps=None):
    """
    Implicit Euler with Newton Gauss Seidel method:

    Outer loop is Newton:
        compute F and J
        solve J delta = -F using Gauss Seidel iterations
        update y = y + delta

    Returns:
        t, Y, newton_iters_per_step, gs_iters_total_per_step, runtime,
        residual_histories_newton, gs_histories
    """
    t0, t1 = float(tspan[0]), float(tspan[1])
    t = np.arange(t0, t1 + h, h)
    n_steps = len(t) - 1

    Y = np.zeros((len(t), 2), dtype=float)
    Y[0] = np.array(y0, dtype=float)

    newton_iters = np.zeros(n_steps, dtype=int)
    gs_total = np.zeros(n_steps, dtype=int)

    residual_histories = {}
    gs_histories = {}

    start = time.time()

    for n in range(n_steps):
        y_old = Y[n].copy()
        t_new = t[n + 1]

        y_new = y_old.copy()  # Newton initial guess
        total_gs_here = 0

        newton_hist = []
        gs_hist = []

        for k in range(newton_max_iter):
            F = residual_implicit_euler(y_new, y_old, t_new, h)
            Fn = float(np.linalg.norm(F, ord=2))
            newton_hist.append(Fn)

            if Fn < newton_tol:
                newton_iters[n] = k + 1
                break

            J = compute_jacobian(y_new, h)

            # Solve J delta = -F using Gauss Seidel iterations
            delta, gs_count, gs_step_hist = gauss_seidel_solve(
                J, -F, tol=gs_tol, max_iter=gs_max_iter, x0=np.zeros(2)
            )
            total_gs_here += gs_count
            gs_hist.append(gs_count)

            # Newton update
            y_new = y_new + delta

            if np.linalg.norm(delta, ord=np.inf) < newton_tol:
                newton_iters[n] = k + 1
                break
        else:
            newton_iters[n] = newton_max_iter

        gs_total[n] = total_gs_here
        Y[n + 1] = y_new

        if store_steps is not None and n in store_steps:
            residual_histories[n] = np.array(newton_hist, dtype=float)
            gs_histories[n] = np.array(gs_hist, dtype=float)

    runtime = time.time() - start
    return t, Y, newton_iters, gs_total, runtime, residual_histories, gs_histories


# ============================================================
# 5) Reference solution and error analysis
# ============================================================
def interpolate_reference(t_ref, Y_ref, t_target):
    """
    Interpolate reference solution to target time points.
    Uses cubic if scipy is available, otherwise linear.
    """
    try:
        from scipy.interpolate import interp1d  # optional
        M_fun = interp1d(t_ref, Y_ref[:, 0], kind="cubic", fill_value="extrapolate")
        T_fun = interp1d(t_ref, Y_ref[:, 1], kind="cubic", fill_value="extrapolate")
        M_i = M_fun(t_target)
        T_i = T_fun(t_target)
        return np.vstack([M_i, T_i]).T
    except Exception:
        M_i = np.interp(t_target, t_ref, Y_ref[:, 0])
        T_i = np.interp(t_target, t_ref, Y_ref[:, 1])
        return np.vstack([M_i, T_i]).T


def compute_reference_solution(tspan, y0, h_ref=0.001):
    """
    High accuracy reference solution using Newton Gauss Seidel at very fine step.
    This is acceptable as a reference because h_ref is extremely small.
    """
    t_ref, Y_ref, _, _, runtime_ref, _, _ = implicit_euler_newton_gauss_seidel(
        tspan, y0, h_ref,
        newton_tol=1e-12, newton_max_iter=40,
        gs_tol=1e-14, gs_max_iter=50
    )
    return t_ref, Y_ref, runtime_ref


def error_metrics(t_test, Y_test, t_ref, Y_ref):
    """
    Compute L2 error and max error for M and T compared to reference.
    """
    Y_ref_on_test = interpolate_reference(t_ref, Y_ref, t_test)
    e = Y_test - Y_ref_on_test

    eM = e[:, 0]
    eT = e[:, 1]

    L2_M = float(np.sqrt(np.mean(eM * eM)))
    L2_T = float(np.sqrt(np.mean(eT * eT)))
    Max_M = float(np.max(np.abs(eM)))
    Max_T = float(np.max(np.abs(eT)))

    return L2_M, L2_T, Max_M, Max_T


def convergence_rates(h_values, err_values):
    """
    Observed convergence rate between successive errors.
    rate_i = log(e_i/e_{i+1}) / log(h_i/h_{i+1})
    """
    rates = [None]
    for i in range(len(h_values) - 1):
        e1 = err_values[i]
        e2 = err_values[i + 1]
        r = np.log(e1 / e2) / np.log(h_values[i] / h_values[i + 1])
        rates.append(float(r))
    return rates


# ============================================================
# 6) Plotting helpers (6 required figures)
# ============================================================
def figure1_trajectories(t, Y_ngs, Y_fp):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(t, Y_ngs[:, 0], label="Newton Gauss Seidel")
    plt.plot(t, Y_fp[:, 0], "--", label="Fixed point")
    plt.title("Moisture content M(t)")
    plt.xlabel("Time")
    plt.ylabel("Moisture")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, Y_ngs[:, 1], label="Newton Gauss Seidel")
    plt.plot(t, Y_fp[:, 1], "--", label="Fixed point")
    plt.title("Temperature T(t)")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.legend()

    plt.tight_layout()
    plt.show()


def figure2_phase_portrait(Y_ngs, Y_fp):
    plt.figure(figsize=(6, 5))
    plt.plot(Y_ngs[:, 0], Y_ngs[:, 1], label="Newton Gauss Seidel")
    plt.plot(Y_fp[:, 0], Y_fp[:, 1], "--", label="Fixed point")
    plt.scatter([Y_ngs[0, 0]], [Y_ngs[0, 1]], label="Start")
    plt.scatter([Y_ngs[-1, 0]], [Y_ngs[-1, 1]], label="End")
    plt.title("Phase portrait T versus M")
    plt.xlabel("Moisture M")
    plt.ylabel("Temperature T")
    plt.legend()
    plt.tight_layout()
    plt.show()


def figure3_loglog_error(h_values, err_values, title):
    plt.figure(figsize=(6, 5))
    plt.loglog(h_values, err_values, "o-", label="Computed error")
    plt.loglog(h_values, [err_values[0] * (h / h_values[0]) for h in h_values], "--", label="Slope 1 reference")
    plt.title(title)
    plt.xlabel("Step size h")
    plt.ylabel("L2 error")
    plt.legend()
    plt.tight_layout()
    plt.show()


def figure4_iteration_comparison(newton_iters, gs_total, fp_iters):
    plt.figure(figsize=(10, 4))
    plt.plot(newton_iters, label="Newton iterations per step")
    plt.plot(fp_iters, label="Fixed point iterations per step")
    plt.title("Iteration comparison per step")
    plt.xlabel("Step index")
    plt.ylabel("Iteration count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(gs_total, label="Total GS iterations per step")
    plt.title("Gauss Seidel effort inside Newton")
    plt.xlabel("Step index")
    plt.ylabel("GS iterations")
    plt.legend()
    plt.tight_layout()
    plt.show()


def figure5_residual_convergence(hist_ngs, hist_fp, sample_steps):
    plt.figure(figsize=(10, 6))

    for s in sample_steps:
        if s in hist_ngs:
            r = hist_ngs[s]
            plt.semilogy(np.arange(1, len(r) + 1), r, label=f"NGS residual step {s}")
        if s in hist_fp:
            r = hist_fp[s]
            plt.semilogy(np.arange(1, len(r) + 1), r, "--", label=f"FP residual step {s}")

    plt.title("Convergence behavior residual norm versus iteration")
    plt.xlabel("Iteration number")
    plt.ylabel("Residual norm")
    plt.legend()
    plt.tight_layout()
    plt.show()


def figure6_cost_accuracy(results_ngs, results_fp):
    plt.figure(figsize=(7, 5))

    t_ngs = [r["time"] for r in results_ngs]
    e_ngs = [r["L2_M"] for r in results_ngs]

    t_fp = [r["time"] for r in results_fp]
    e_fp = [r["L2_M"] for r in results_fp]

    plt.scatter(t_ngs, e_ngs, label="Newton Gauss Seidel")
    plt.scatter(t_fp, e_fp, label="Fixed point")

    plt.yscale("log")
    plt.title("Cost accuracy trade off")
    plt.xlabel("CPU time (s)")
    plt.ylabel("L2 error in M")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 7) Tables
# ============================================================
def table1_solver_performance(h, newton_iters, gs_total, time_ngs, fp_iters, time_fp):
    total_newton = int(np.sum(newton_iters))
    total_gs = int(np.sum(gs_total))
    total_fp = int(np.sum(fp_iters))

    avg_newton = float(np.mean(newton_iters))
    avg_gs_per_newton_step = float(total_gs / max(total_newton, 1))
    avg_fp = float(np.mean(fp_iters))

    time_per_iter_ngs_ms = 1000.0 * time_ngs / max(total_newton + total_gs, 1)
    time_per_iter_fp_ms = 1000.0 * time_fp / max(total_fp, 1)

    print("\nTable 1: Solver performance comparison at h =", h)
    print("Metric                          | Newton Gauss Seidel        | Fixed point")
    print("--------------------------------|----------------------------|----------------------------")
    print(f"Avg Newton iter per step         | {avg_newton:>10.3f}                 | N A")
    print(f"Avg GS iter per Newton step      | {avg_gs_per_newton_step:>10.3f}                 | N A")
    print(f"Total GS iterations              | {total_gs:>10d}                 | N A")
    print(f"Avg fixed point iter per step    | N A                        | {avg_fp:>10.3f}")
    print(f"Total iterations                 | {total_newton + total_gs:>10d}                 | {total_fp:>10d}")
    print(f"CPU time (seconds)               | {time_ngs:>10.6f}                 | {time_fp:>10.6f}")
    print(f"Time per iteration (ms)          | {time_per_iter_ngs_ms:>10.4f}                 | {time_per_iter_fp_ms:>10.4f}")


def table2_accuracy(h_values, rows, rates_M, rates_T, solver_name):
    print("\nTable 2: Accuracy analysis for", solver_name)
    print("Step size h | L2 Error M  | L2 Error T  | Max Error M | Max Error T | Order M | Order T")
    print("-----------|-------------|-------------|-------------|-------------|---------|--------")

    for i, r in enumerate(rows):
        om = "-" if rates_M[i] is None else f"{rates_M[i]:.2f}"
        ot = "-" if rates_T[i] is None else f"{rates_T[i]:.2f}"
        print(f"{r['h']:>9.2f} | {r['L2_M']:<11.3e} | {r['L2_T']:<11.3e} | {r['Max_M']:<11.3e} | {r['Max_T']:<11.3e} | {om:>7} | {ot:>6}")


def table3_efficiency(h_values, res_ngs, res_fp):
    print("\nTable 3: Computational efficiency")
    print("Step size | NGS Time (s) | FP Time (s)  | NGS Error M | FP Error M  | Winner")
    print("---------|--------------|--------------|------------|------------|--------")

    for r_ngs, r_fp in zip(res_ngs, res_fp):
        winner = "NGS" if (r_ngs["L2_M"] / max(r_ngs["time"], 1e-15)) < (r_fp["L2_M"] / max(r_fp["time"], 1e-15)) else "FP"
        print(f"{r_ngs['h']:>7.2f} | {r_ngs['time']:<12.6f} | {r_fp['time']:<12.6f} | {r_ngs['L2_M']:<10.3e} | {r_fp['L2_M']:<10.3e} | {winner}")


# ============================================================
# 8) Main experiments runner
# ============================================================
def run_experiments():
    tspan = (0.0, 50.0)
    y0 = (0.6, 20.0)

    # ---------------------------------------
    # Experiment 1: Solver comparison at h=0.1
    # ---------------------------------------
    h_main = 0.1
    sample_steps = [5, 200, 450]

    t_ngs, Y_ngs, newton_iters, gs_total, time_ngs, hist_ngs, _ = implicit_euler_newton_gauss_seidel(
        tspan, y0, h_main,
        newton_tol=1e-10, newton_max_iter=30,
        gs_tol=1e-12, gs_max_iter=20,
        store_steps=sample_steps
    )

    t_fp, Y_fp, fp_iters, time_fp, hist_fp = implicit_euler_fixed_point(
        tspan, y0, h_main, tol=1e-10, max_iter=80, store_steps=sample_steps
    )

    if not np.allclose(t_ngs, t_fp):
        raise RuntimeError("Time grids do not match for h=0.1 comparison")

    # Figures 1, 2, 4, 5
    figure1_trajectories(t_ngs, Y_ngs, Y_fp)
    figure2_phase_portrait(Y_ngs, Y_fp)
    figure4_iteration_comparison(newton_iters, gs_total, fp_iters)
    figure5_residual_convergence(hist_ngs, hist_fp, sample_steps)

    # Table 1
    table1_solver_performance(h_main, newton_iters, gs_total, time_ngs, fp_iters, time_fp)

    # ---------------------------------------
    # Experiment 2: Accuracy validation
    # ---------------------------------------
    t_ref, Y_ref, time_ref = compute_reference_solution(tspan, y0, h_ref=0.001)
    print("\nReference solution computed with h_ref = 0.001 runtime =", f"{time_ref:.4f}", "seconds")

    h_values = [0.2, 0.1, 0.05, 0.01]

    # run solvers at multiple step sizes and compute errors
    res_ngs = []
    res_fp = []

    for h in h_values:
        # NGS
        t1, Y1, ni, gs, tm1, _, _ = implicit_euler_newton_gauss_seidel(
            tspan, y0, h,
            newton_tol=1e-10, newton_max_iter=30,
            gs_tol=1e-12, gs_max_iter=20
        )
        L2_M, L2_T, Max_M, Max_T = error_metrics(t1, Y1, t_ref, Y_ref)
        res_ngs.append({
            "h": h, "L2_M": L2_M, "L2_T": L2_T, "Max_M": Max_M, "Max_T": Max_T,
            "time": tm1, "avg_newton": float(np.mean(ni)), "avg_gs": float(np.mean(gs))
        })

        # FP
        t2, Y2, it2, tm2, _ = implicit_euler_fixed_point(tspan, y0, h, tol=1e-10, max_iter=80)
        L2_M, L2_T, Max_M, Max_T = error_metrics(t2, Y2, t_ref, Y_ref)
        res_fp.append({
            "h": h, "L2_M": L2_M, "L2_T": L2_T, "Max_M": Max_M, "Max_T": Max_T,
            "time": tm2, "avg_fp": float(np.mean(it2))
        })

    # Convergence rates (should be near 1)
    rates_ngs_M = convergence_rates(h_values, [r["L2_M"] for r in res_ngs])
    rates_ngs_T = convergence_rates(h_values, [r["L2_T"] for r in res_ngs])
    rates_fp_M = convergence_rates(h_values, [r["L2_M"] for r in res_fp])
    rates_fp_T = convergence_rates(h_values, [r["L2_T"] for r in res_fp])

    # Table 2
    table2_accuracy(h_values, res_ngs, rates_ngs_M, rates_ngs_T, "Newton Gauss Seidel")
    table2_accuracy(h_values, res_fp, rates_fp_M, rates_fp_T, "Fixed point")

    # Figure 3 log log error vs h (use NGS as example)
    figure3_loglog_error(h_values, [r["L2_M"] for r in res_ngs], "Convergence order check for M using Newton Gauss Seidel")
    figure3_loglog_error(h_values, [r["L2_T"] for r in res_ngs], "Convergence order check for T using Newton Gauss Seidel")

    # ---------------------------------------
    # Experiment 4: Cost accuracy trade off
    # ---------------------------------------
    figure6_cost_accuracy(res_ngs, res_fp)

    # Table 3
    table3_efficiency(h_values, res_ngs, res_fp)

    print("\nExpected finding")
    print("Implicit Euler should show first order convergence so rates should be close to 1")


if __name__ == "__main__":
    run_experiments()
