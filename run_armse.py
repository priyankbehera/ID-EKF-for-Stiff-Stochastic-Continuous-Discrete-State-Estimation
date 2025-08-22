import os, numpy as np, matplotlib.pyplot as plt
from models import dahlquist_f, dahlquist_J, dahlquist_h, dahlquist_H, vdp_f, vdp_J, vdp_h, vdp_H
from discretization import simulate_sde_it15
from filters import DiscreteModel, EKF, UKF, CKF, IDEKF

os.makedirs("results", exist_ok=True)

def build_discrete_model(f, Jf, G, dt):
    def g(x):
        dx = f(x, 0.0)
        J = Jf(x, 0.0)
        return x + dx*dt + 0.5*(J @ dx)*(dt**2)
    def F(x):
        return np.eye(x.shape[0]) + Jf(x, 0.0)*dt
    Qd = G @ G.T * dt
    return DiscreteModel(g, F, Qd)

def run_once(model_name: str, dt_list, N_runs=20, seed=0):
    rng0 = np.random.default_rng(seed)
    results = {}
    for dt in dt_list:
        if model_name == "dahlquist":
            mu, j = -1e4, 3
            f, Jf = dahlquist_f(mu, j), dahlquist_J(mu, j)
            G = np.array([[1.0]])
            x0 = np.array([1.0])
            t0, tf = 0.0, 4.0
            h, H = dahlquist_h(), dahlquist_H()
            R = np.array([[0.04]])
        elif model_name == "vdp":
            mu = 1e4
            f, Jf = vdp_f(mu), vdp_J(mu)
            G = np.array([[0.0, 0.0],
                          [0.0, 1.0]])
            x0 = np.array([2.0, 0.0])
            t0, tf = 0.0, 2.0
            h, H = vdp_h(), vdp_H()
            R = np.array([[0.04]])
        else:
            raise ValueError("unknown model")

        dm = build_discrete_model(f, Jf, G, dt)

        T_steps = int(np.ceil((tf - t0)/dt)) + 1
        trajs_true = np.zeros((N_runs, T_steps, x0.shape[0]))
        trajs_est  = {k: np.zeros((N_runs, T_steps, x0.shape[0])) for k in ["EKF","UKF","CKF","IDEKF"]}

        for i in range(N_runs):
            rng = np.random.default_rng(rng0.integers(1<<32))
            ts, xs = simulate_sde_it15(x0, t0, tf, dt, f, Jf, G, rng, refine=100)
            if not np.all(np.isfinite(xs)):                                     
                continue
            zs = np.zeros((xs.shape[0], R.shape[0]))
            for k in range(xs.shape[0]):
                zs[k] = h(xs[k]) + np.linalg.cholesky(R) @ rng.normal(size=R.shape[0])
            trajs_true[i] = xs

            x_init = x0 + 0.1*rng.normal(size=x0.shape[0])
            P_init = np.eye(x0.shape[0]) * 1e-2

            ekf  = EKF(dm, h, H, R)
            ukf  = UKF(dm, h, R)
            ckf  = CKF(dm, h, R)
            idek = IDEKF(dm, h, H, R)

            Xs = {k: np.zeros_like(xs) for k in trajs_est.keys()}
            Ps = {k: np.zeros((xs.shape[0], x0.shape[0], x0.shape[0])) for k in trajs_est.keys()}
            for name, ftr in zip(["EKF","UKF","CKF","IDEKF"], [ekf,ukf,ckf,idek]):
                x, P = x_init.copy(), P_init.copy()
                Xs[name][0], Ps[name][0] = x.copy(), P.copy()
            for k in range(1, xs.shape[0]):
                z = zs[k]
                for name, ftr in zip(["EKF","UKF","CKF","IDEKF"], [ekf,ukf,ckf,idek]):
                    x_pred, P_pred = ftr.predict(Xs[name][k-1], Ps[name][k-1])
                    x_upd, P_upd, _, _ = ftr.update(x_pred, P_pred, z)
                    Xs[name][k], Ps[name][k] = x_upd, P_upd
            for name in trajs_est.keys():
                trajs_est[name][i] = Xs[name]

        results[dt] = {}
        for name in trajs_est.keys():                                              # <<< NEW
            err = trajs_est[name] - trajs_true                                    # <<< NEW
            err2 = np.sum(err**2, axis=2)                                         # (runs, T)  <<< NEW
            results[dt][name] = float(np.sqrt(np.nanmean(err2)))                  # <<< NEW
        valid_runs = int(np.isfinite(trajs_true).all(axis=(1,2)).sum())           # <<< NEW
        print(f"dt={dt}: valid runs = {valid_runs}/{N_runs}") 
        
        labels = list(trajs_est.keys())
        vals = [results[dt][lab] for lab in labels]
        import csv
        csv_path = f"results/{model_name}_armse.csv"
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["dt"] + labels)
            w.writerow([dt] + vals)

        import matplotlib
        plt.figure()
        plt.bar(labels, vals)
        plt.ylabel("ARMSE")
        plt.title(f"{model_name.upper()} — ARMSE (dt={dt})")
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_armse_dt{dt:.3g}.png", dpi=150)
        plt.close()

    return results

if __name__ == "__main__":
    dts = [0.05, 0.1, 0.2, 0.5]
    print("Running ARMSE on Dahlquist...")
    run_once("dahlquist", dts, N_runs=20, seed=0)
    print("Running ARMSE on Van der Pol...")
    run_once("vdp", dts, N_runs=20, seed=1)
    print("Done. Results in ./results/*.csv and *.png")
