from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggingCallback(BaseCallback):
    """
    Log les récompenses par épisode dans TensorBoard.
    Compatible avec PPO (SB3).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = 0.0
        self.episode_length = 0

    def _on_step(self) -> bool:
        # Reward du pas courant
        reward = float(self.locals["rewards"][0])
        self.episode_rewards += reward
        self.episode_length += 1

        # Fin d'épisode détectée
        if self.locals["dones"][0]:
            # Reward moyen par épisode
            ep_rew_mean = self.episode_rewards

            # Log TensorBoard (sous rollout/)
            self.logger.record("rollout/ep_rew_mean", ep_rew_mean)
            self.logger.record("rollout/ep_len", self.episode_length)

            # Reset
            self.episode_rewards = 0.0
            self.episode_length = 0

        return True