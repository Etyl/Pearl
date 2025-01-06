from __future__ import annotations

import numpy as np
from collections import deque, defaultdict
from typing import Optional, Tuple, Dict, List, Type, Deque
import functools
import random
import torch
from sklearn.linear_model import LinearRegression
from scipy.special import softmax
from torch import Tensor

from pearl.api import SubjectiveState, Action, Reward, ActionSpace
from pearl.replay_buffers import BasicReplayBuffer, ReplayBuffer, TransitionBatch, Transition, TensorBasedReplayBuffer
from pearl.utils.device import get_default_device

Traj: Type = Dict[str, List[float]]


def AW(trajs: List[Traj]) -> Dict[str, np.ndarray[float]]:
  ep_rets = np.asarray(list(map(lambda traj: np.sum(traj["reward"]), trajs)))
  ep_lens =  np.asarray(list(map(lambda traj: len(traj["observation"]), trajs)))
  s0s = np.array([traj["observation"][0] for traj in trajs])
  v = LinearRegression().fit(s0s, ep_rets).predict(s0s)
  weights = np.asarray(functools.reduce(lambda a, b: a + b,
              [[w] * l for w, l in zip((ep_rets - v), ep_lens)]))
  weights = (weights - weights.min()) / (weights.max() - weights.min())
  dataset = {k: np.concatenate([traj[k] for traj in trajs], axis=0) for k in trajs[0].keys()}
  dataset["weights"] = weights
  return dataset

def RR(trajs: List[Traj]) -> Dict[str, np.ndarray[float]]:
    """
    Based on return resampling
    """
    weights = np.asarray(list(map(lambda traj: traj["score"][0], trajs)))
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    dataset = {k: np.concatenate([traj[k] for traj in trajs], axis=0) for k in trajs[0].keys()}
    dataset["weights"] = [[w]*len(traj["observation"]) for w,traj in zip(weights,trajs)]
    del dataset["score"]
    return dataset


class WeightedReplayBuffer(TensorBasedReplayBuffer):
    def __init__(self, temperature: float) -> None:
        super().__init__(0)
        self.temperature = temperature
        self._memory: List[Transition] = []
        self._weights: List[float] = []
        self._device_for_batches: torch.device = get_default_device()
        self._sample_dict: Dict[str, List[Traj]] = defaultdict(list)

    @property
    def device_for_batches(self) -> torch.device:
        return self._device_for_batches
    
    @device_for_batches.setter
    def device_for_batches(self, new_device_for_batches: torch.device) -> None:
        self._device_for_batches = new_device_for_batches

    def __len__(self):
        return len(self._memory)

    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        terminated: bool,
        truncated: bool,
        curr_available_actions: ActionSpace | None = None,
        next_state: SubjectiveState | None = None,
        next_available_actions: ActionSpace | None = None,
        max_number_actions: int | None = None,
        cost: float | None = None,
    ) -> None:
        raise "Method should not be called, use add_traj instead"

    def _store_transition(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        terminated: bool,
        truncated: bool,
        curr_available_actions_tensor_with_padding: Tensor | None,
        curr_unavailable_actions_mask: Tensor | None,
        next_state: SubjectiveState | None,
        next_available_actions_tensor_with_padding: Tensor | None,
        next_unavailable_actions_mask: Tensor | None,
        cost: float | None = None,
    ) -> None:
        transition = Transition(
            state=self._process_non_optional_single_state(state),
            action=self._process_single_action(action),
            reward=self._process_single_reward(reward),
            next_state=self._process_single_state(next_state),
            curr_available_actions=curr_available_actions_tensor_with_padding,
            curr_unavailable_actions_mask=curr_unavailable_actions_mask,
            next_available_actions=next_available_actions_tensor_with_padding,
            next_unavailable_actions_mask=next_unavailable_actions_mask,
            terminated=self._process_single_terminated(terminated),
            truncated=self._process_single_truncated(truncated),
            cost=self._process_single_cost(cost),
        )
        self._memory.append(transition)



    def sample(self, batch_size: int) -> TransitionBatch:
        """
        The shapes of input and output are:
        input: batch_size

        output: TransitionBatch(
          state = tensor(batch_size, state_dim),
          action = tensor(batch_size, action_dim),
          reward = tensor(batch_size, ),
          next_state = tensor(batch_size, state_dim),
          curr_available_actions = tensor(batch_size, action_dim, action_dim),
          curr_available_actions_mask = tensor(batch_size, action_dim),
          next_available_actions = tensor(batch_size, action_dim, action_dim),
          next_available_actions_mask = tensor(batch_size, action_dim),
          terminated = tensor(batch_size, ),
          truncated = tensor(batch_size, ),
        )
        """
        if batch_size > len(self):
            raise ValueError(
                f"Can't get a batch of size {batch_size} from a replay buffer with "
                f"only {len(self)} elements"
            )
        samples = random.choices(self._memory, weights=self._weights, k=batch_size)
        return self._create_transition_batch(
            # pyre-fixme[6]: For 1st argument expected `List[Transition]` but got
            #  `List[Union[Transition, TransitionBatch]]`.
            transitions=samples,
            is_action_continuous=self._is_action_continuous,
        )

    def clear(self) -> None:
        self._memory = deque([], maxlen=self.capacity)
        self._weights = deque([], maxlen=self.capacity)
        self._sample_dict = defaultdict(list)

    def add_trajs(self, trajs: List[Traj], group_name: str) -> None:
        self._sample_dict[group_name] = self._sample_dict[group_name] + trajs

    def add_traj(self, traj: Traj, group_name: str) -> None:
        self._sample_dict[group_name].append(traj)

    def build_memory(self) -> None:
        for trajs in self._sample_dict.values():
            dataset = RR(trajs)

            for i in range(len(dataset["observation"])):
                super().push(
                    state = dataset["observation"][i],
                    action = dataset["action"][i],
                    reward = dataset["reward"][i],
                    next_state = dataset["next_observation"][i],
                    terminated = dataset["terminated"][i],
                    truncated = dataset["truncated"][i],
                    curr_available_actions=dataset["curr_available_actions"][i],
                    next_available_actions=dataset["next_available_actions"][i],
                )
            # self._weights += list(softmax(dataset["weights"] / self.temperature))
            self._weights += list(dataset["weights"])


