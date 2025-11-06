import numpy as np

def _softmax_stable(log_w: np.ndarray) -> np.ndarray:
    m = np.max(log_w)
    z = np.exp(log_w - m)
    s = z.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(log_w) / log_w.size
    p = z / s
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    s2 = p.sum()
    return p if s2 == 1.0 or s2 == 0.0 else (p / s2)

class MWUSolver:
    def __init__(self, A: np.ndarray, eta_def: float = 0.05, eta_att: float = 0.05, seed: int = 7):
        # Keep a scaled copy of A to keep updates in a safe range
        A = A.astype(np.float64)
        scale = np.max(np.abs(A))
        self.scale = max(1.0, float(scale))
        self.A = A / self.scale
        self.rng = np.random.default_rng(seed)
        self.eta_d = float(eta_def)
        self.eta_a = float(eta_att)

    def run(self, T: int = 20000):
        R, C = self.A.shape
        # log-weights avoid overflow
        log_w_d = np.zeros(R, dtype=np.float64)
        log_w_a = np.zeros(C, dtype=np.float64)

        pay_hist = []
        xd_hist = []
        qa_hist = []

        for t in range(1, T + 1):
            x = _softmax_stable(log_w_d)
            q = _softmax_stable(log_w_a)

            # safety: uniform fallback if something funky sneaks in
            if not np.isfinite(x.sum()) or x.sum() <= 0:
                x = np.ones(R) / R
            if not np.isfinite(q.sum()) or q.sum() <= 0:
                q = np.ones(C) / C

            H_idx = self.rng.choice(R, p=x)
            j_idx = self.rng.choice(C, p=q)

            payoff = self.A[H_idx, j_idx] * self.scale  # record in original units
            pay_hist.append(payoff)

            # FULL-INFO estimates
            est_def = self.A[:, j_idx]              # defender maximizes A
            est_att = -self.A[H_idx, :]             # attacker maximizes -A

            # log-weight updates (no exp here!)
            log_w_d += self.eta_d * est_def
            log_w_a += self.eta_a * est_att

            # keep numbers tame
            if t % 100 == 0:
                log_w_d -= np.max(log_w_d)
                log_w_a -= np.max(log_w_a)

            xd_hist.append(x)
            qa_hist.append(q)

        x_bar = np.mean(np.vstack(xd_hist), axis=0)
        q_bar = np.mean(np.vstack(qa_hist), axis=0)
        return {
            "x_bar": x_bar,
            "q_bar": q_bar,
            "pay_hist": np.array(pay_hist, dtype=np.float64),
        }
