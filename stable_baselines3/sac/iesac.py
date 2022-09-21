from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.buffers import PERReplayBuffer
from stable_baselines3.common.noise import ActionNoise

from stable_baselines3.sac.sac import SAC
from torch.nn import functional as F
import torch as th
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, GradientCallback
from stable_baselines3.common.utils import polyak_update


class IESAC(SAC):
    replay_buffer: PERReplayBuffer

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[PERReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        bootstap_entropy: float = 0.25,
        max_bootstrap=10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class if replay_buffer_class else PERReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.bootstrap_entropy = bootstap_entropy
        self.max_bootstrap = max_bootstrap

    def train(self, gradient_steps: int, batch_size: int = 64, callback: BaseCallback = None) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            log_prob_prio = None

            with th.no_grad():
                needs_next_step = np.ones_like(replay_data.idxs, dtype=bool)
                next_obs = replay_data.next_observations
                rewards = replay_data.rewards
                dones = replay_data.dones
                idxs = replay_data.idxs
                step = 1
                entropy_target = None

                target_q_values = replay_data.rewards

                while needs_next_step.any():

                    if step < self.max_bootstrap:
                        # Log prob of next observation
                        action = self.actor.forward(next_obs, deterministic=True)
                        det_log_prob = self.actor.action_dist.log_prob(
                            action, self.actor.action_dist.gaussian_actions)
                        if log_prob_prio is None:
                            log_prob_prio = det_log_prob
                        if entropy_target is None:
                            # Compute entropy target, entropy should be lower than bootstrap_entropy percentile of the batch
                            entropy_target = np.percentile(det_log_prob.detach().cpu().numpy(), self.bootstrap_entropy * 100)

                        # Is entropy too high in the current step index
                        batch_next_step = det_log_prob > entropy_target & ~dones
                        # Is entropy too high in the initial step index
                        needs_next_step[needs_next_step] = batch_next_step
                    else:
                        batch_next_step = np.zeros_like(idxs, dtype=bool)
                        needs_next_step.fill(False)

                    # Q-Function accurate -> bootstrap to critic

                    next_obs = next_obs[~batch_next_step]
                    # Select action according to policy
                    next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                    # Compute the next Q values: min over all critics targets
                    next_q_values = th.cat(self.critic_target(next_obs, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    # add entropy term
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    # td error + entropy term
                    # BUG: I NEED ONLY THE NOT NEEDS NEXT STEP FOR THE CURRENT BATCH HERE OR WE KEEP BOOTSTRAPPING EVERY TIME
                    target_q_values[~needs_next_step] += (1 - dones) * self.gamma ** step * next_q_values

                    # Q-Function inaccurate -> add next reward and continue

                    # Get data for the next step
                    new_data = self.replay_buffer.get_next_step(idxs[batch_next_step])
                    next_obs = new_data.next_observations
                    rewards = new_data.rewards
                    dones = new_data.dones
                    idxs = new_data.idxs
                    # Add discounted reward to target and continue
                    target_q_values[needs_next_step] += self.gamma ** step * rewards

                    step += 1

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update buffer priorities
            log_prob_prio = log_prob_prio.detach().cpu().numpy()
            # Shift logprob from -1 to inf to 0 to inf
            log_prob_prio = np.log(1 + np.exp(log_prob_prio))
            self.replay_buffer.update_priorities(
                replay_data.idxs, log_prob_prio)

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            # Call callback
            if isinstance(callback, GradientCallback):
                callback.update_locals(locals())
                callback._on_gradient_step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
