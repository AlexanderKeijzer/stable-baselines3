import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import PERReplayBuffer, CountedReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, GradientCallback
from stable_baselines3 import DQN
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, is_vectorized_observation, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy


class PDQN(DQN):
    replay_buffer: PERReplayBuffer
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[PERReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        bootstap_percentile: float = 0.75,
        max_bootstrap=10,
        prio_overwrite_func: Optional[Callable[["PDQN", th.Tensor], np.ndarray]] = None,
        bootstrap_overwrite_func: Optional[Callable[["PDQN", th.Tensor], th.Tensor]] = None,
        bootstrap_target_overwrite: Optional[float] = None,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        self.bootstrap_percentile = bootstap_percentile
        self.max_bootstrap = max_bootstrap
        self.prio_overwrite_func = prio_overwrite_func
        self.bootstrap_overwrite_func = bootstrap_overwrite_func
        self.bootstrap_target_overwrite = bootstrap_target_overwrite

    def train(self, gradient_steps: int, batch_size: int = 100, callback: BaseCallback = None) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            bootstrap_distance = th.zeros_like(replay_data.rewards, dtype=th.int8)
            bootstrap_states = th.tensor([], dtype=replay_data.next_observations.dtype, device=self.device)

            with th.no_grad():
                # Did we not already bootstrap:
                needs_next_step = th.ones_like(replay_data.rewards, dtype=bool)
                # Do we bootstrap this step
                ends_this_step = th.zeros_like(replay_data.rewards, dtype=bool)
                next_obs = replay_data.next_observations
                rewards = replay_data.rewards
                dones = replay_data.dones
                # Index in the buffer
                idxs = replay_data.idxs
                step = 1
                # The target log prob to allow bootstraps (can be overwritten)
                log_prob_target = self.bootstrap_target_overwrite

                # The target q value = r1
                target_q_values = replay_data.rewards.clone()

                # Is there any sample that didn't bootstrap yet
                while needs_next_step.any():

                    # Are we not at the max horizon
                    if step < self.max_bootstrap:
                        # Log prob of next observation
                        if self.bootstrap_overwrite_func is None:
                            action = self.actor.forward(next_obs, deterministic=True)
                            det_log_prob = self.actor.action_dist.log_prob(
                                action, self.actor.action_dist.gaussian_actions)
                        else:
                            det_log_prob = self.bootstrap_overwrite_func(self, next_obs)

                        # Only done once for the initial step (or overwitten)
                        if log_prob_target is None:
                            # Compute entropy target, entropy should be lower than bootstrap_entropy percentile of the batch
                            log_prob_target = th.quantile(det_log_prob, self.bootstrap_percentile)

                        # Is entropy too high in the current step batch index
                        batch_next_step = (det_log_prob < log_prob_target) & ~dones.flatten().bool()

                        # Is entropy too high in the initial batch index
                        ends_this_step.zero_()
                        ends_this_step[needs_next_step] = ~batch_next_step

                        # In the initial batch set the needs next the to false if we bootstrap this step
                        needs_next_step[needs_next_step.clone()] = batch_next_step
                    else:
                        # If we are at the max horizon, bootstrap everything
                        batch_next_step = th.zeros_like(rewards, dtype=bool).squeeze()
                        ends_this_step.zero_()
                        ends_this_step[needs_next_step] = True
                        needs_next_step.zero_()

                    # Q-Function accurate -> bootstrap to critic

                    # Select next observations where we wil currently bootstrap
                    next_obs = next_obs[~batch_next_step]
                    curr_dones = dones[~batch_next_step]

                    # STANDARD DQN (except for the discount factor ** step)
                    ##############
                    # Compute the next Q-values using the target network
                    next_q_values = self.q_net_target(replay_data.next_observations)
                    # Follow greedy policy: use the one with the highest value
                    next_q_values, _ = next_q_values.max(dim=1)
                    # Avoid potential broadcast issue
                    next_q_values = next_q_values.reshape(-1, 1)
                    target_q_values[ends_this_step] += ((1 - curr_dones) * self.gamma ** step * next_q_values).squeeze()
                    ##############

                    # Purely for logging
                    dist = th.zeros_like(curr_dones, dtype=th.int8).flatten()
                    dist[curr_dones.bool().flatten()] = -1
                    dist[~curr_dones.bool().flatten()] = step
                    bootstrap_distance[ends_this_step] = dist
                    bootstrap_states = th.cat((bootstrap_states, next_obs[~curr_dones.bool()]), dim=0)

                    # Increase count for these indices if we use a counted replay buffer
                    if isinstance(self.replay_buffer, CountedReplayBuffer):
                        self.replay_buffer.increase_count(idxs[~batch_next_step.cpu().numpy()].flatten())

                    # Q-Function inaccurate -> add next reward and continue
                    # !So for all the samples that we did not bootstrap this step!

                    # Get data for the next step
                    new_data = self.replay_buffer.get_next_step(idxs[batch_next_step.cpu().numpy()])
                    next_obs = new_data.next_observations
                    rewards = new_data.rewards
                    dones = new_data.dones
                    idxs = new_data.idxs
                    # Add discounted reward to target and continue
                    target_q_values[needs_next_step] += (self.gamma ** step * rewards).squeeze()

                    step += 1

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Update buffer priorities
            if not isinstance(self.replay_buffer, CountedReplayBuffer):
                with th.no_grad():
                    if self.prio_overwrite_func is None:
                        action = self.actor.forward(replay_data.observations, deterministic=True)
                        log_prob_prio = self.actor.action_dist.log_prob(
                            action, self.actor.action_dist.gaussian_actions)
                        log_prob_prio = log_prob_prio.cpu().numpy()
                        # Shift logprob from -1 to inf to 0 to inf
                        log_prob_prio = np.log(1 + np.exp(log_prob_prio))
                    else:
                        log_prob_prio = self.prio_overwrite_func(self, replay_data.observations)
                    self.replay_buffer.update_priorities(
                        replay_data.idxs, log_prob_prio)

            # Call callback
            if isinstance(callback, GradientCallback):
                callback.update_locals(locals())
                callback._on_gradient_step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state
