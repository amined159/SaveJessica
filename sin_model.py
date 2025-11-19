import numpy as np

class OnlineSinusoidModel:
    """
    Online learner for:
        y ≈ c + C cos(ω t) + S sin(ω t)

    - ω (frequency) is fixed.
    - θ = [c, C, S]^T is updated online via RLS.
    """

    def __init__(
        self,
        period: float,
        init_offset: float = 0.3,
        init_amplitude: float = 0.05,
        init_phase: float = 0.0,
        init_cov: float = 1.0,
        forgetting: float = 0.99,
    ):
        self.period = period
        self.omega = 2 * np.pi / period
        self.forgetting = forgetting

        # Convert offset + amplitude + phase into (c, C, S)
        c = init_offset
        A = init_amplitude
        phi = init_phase

        C = A * np.cos(phi)
        S = -A * np.sin(phi)

        self.theta = np.array([c, C, S], dtype=float)  # [c, C, S]
        self.P = np.eye(3) * init_cov                  # uncertainty on θ

    def _phi(self, t: int) -> np.ndarray:
        """Feature vector φ(t) = [1, cos(ωt), sin(ωt)]."""
        angle = self.omega * t
        return np.array([1.0, np.cos(angle), np.sin(angle)], dtype=float)

    def predict(self, t: int) -> float:
        """Predict survival probability at time t."""
        phi_t = self._phi(t)
        y_hat = float(phi_t @ self.theta)
        # clip to [0,1] since it's a probability
        return float(np.clip(y_hat, 0.0, 1.0))

    def update(self, t: int, survived: int, morties_sent: int) -> None:
        """Update parameters using observed outcome at trip t."""
        if morties_sent <= 0:
            return

        y = survived / morties_sent  # empirical survival in [0,1]
        phi_t = self._phi(t)

        # RLS / Kalman update
        P = self.P
        θ = self.theta
        λ = self.forgetting

        # Prediction
        y_hat = float(phi_t @ θ)
        e = y - y_hat  # scalar

        denom = λ + float(phi_t @ P @ phi_t)
        K = (P @ phi_t) / denom  # shape (3,)

        θ_new = θ + K * e
        P_new = (1.0 / λ) * (P - np.outer(K, phi_t) @ P)

        self.theta = θ_new
        self.P = P_new
    def get_params(self):
        """
        Return current (offset, amplitude, phase_rad, phase_deg, C, S)
        derived from theta = [c, C, S].
        """
        c, C, S = self.theta
        amplitude = float(np.sqrt(C**2 + S**2))
        # recall: C = A cos φ, S = -A sin φ -> φ = atan2(-S, C)
        phase_rad = float(np.arctan2(-S, C))
        phase_deg = float(np.degrees(phase_rad))
        return {
            "offset": float(c),
            "amplitude": amplitude,
            "phase_rad": phase_rad,
            "phase_deg": phase_deg,
            "C": float(C),
            "S": float(S),
        }
        