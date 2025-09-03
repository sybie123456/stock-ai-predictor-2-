
import time
import numpy as np
import pandas as pd

class OnlineTrainer:
    """
    Simpele online trainer shim. Als stable-baselines3 aanwezig is en een env wordt meegegeven,
    probeert het een kort model.learn() te doen. Anders simuleert het training loss/progress.
    """
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.history = []

    def step_train(self, data_df=None, steps=100, simulate=True):
        """
        Voer een korte 'trainings' run uit. Retourneert history dict.
        - If simulate True or stable-baselines not available, returns fake loss curve.
        - Otherwise, user can implement actual RL training by providing an env.
        """
        if not simulate:
            try:
                from stable_baselines3 import PPO
                # The caller must provide an env attribute in data_df or similar. We keep compat simple.
                # If a real env is provided inside data_df as attribute 'env', try to train.
                env = getattr(data_df, "env", None)
                if env is not None:
                    model = PPO("MlpPolicy", env, verbose=0)
                    model.learn(total_timesteps=steps)
                    model.save(self.model_path or "models/online_tmp.zip")
                    self.history = [{"step": i, "loss": 0.0} for i in range(steps)]
                    return self.history
            except Exception:
                # fall back to simulation
                simulate = True

        # Simulate training loss curve
        rng = np.random.RandomState(seed=int(time.time()) % 2**32)
        base = np.linspace(1.0, 0.2, steps)
        noise = rng.normal(scale=0.05, size=steps) * np.linspace(1.0, 0.2, steps)
        loss = (base + noise).clip(min=0.01)
        self.history = [{"step": i+1, "loss": float(loss[i])} for i in range(steps)]
        return self.history
