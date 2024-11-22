# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch

from pearl.api.space import Space
from pearl.utils.instantiations.spaces.utils import reshape_to_1d_tensor
from torch import Tensor

try:
    import gymnasium as gym
    from gymnasium.spaces import Box

    logging.info("Using 'gymnasium' package.")
except ModuleNotFoundError:
    import gym
    from gym.spaces import Box

    logging.warning("Using deprecated 'gym' package.")


class BoxSpace(Space):
    """A continuous, box space. This class is a wrapper around Gymnasium's
    `Box` space, but uses PyTorch tensors instead of NumPy arrays."""

    def __init__(
        self,
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        low: float | np.ndarray | Tensor,
        # pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
        high: float | np.ndarray | Tensor,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """Constructs a `BoxSpace`.

        Args:
            low: The lower bound on each dimension of the space.
            high: The upper bound on each dimension of the space.
            seed: Random seed used to initialize the random number generator of the
                underlying Gymnasium `Box` space.
        """
        # pyre-fixme[9]: low has type `Union[float, Tensor]`; used as `ndarray[Any,
        #  Any]`.
        low = low.numpy(force=True) if isinstance(low, Tensor) else low
        # pyre-fixme[9]: high has type `Union[float, Tensor]`; used as `ndarray[Any,
        #  Any]`.
        high = high.numpy(force=True) if isinstance(high, Tensor) else high
        self._gym_space = Box(low=low, high=high, seed=seed)

    @property
    def is_continuous(self) -> bool:
        """Checks whether this is a continuous space."""
        return True

    def sample(self, mask: Tensor | None = None) -> Tensor:
        """Sample an element uniformly at random from the space.

        Args:
            mask: An unused argument for the case of a `BoxSpace`, which
                does not support masking.

        Returns:
            A randomly sampled element.
        """
        if mask is not None:
            logging.warning("Masked sampling is not supported in `BoxSpace`. Ignoring.")
        return torch.from_numpy(self._gym_space.sample())

    @property
    def low(self) -> Tensor:
        """Returns the lower bound of the space."""
        return reshape_to_1d_tensor(torch.from_numpy(self._gym_space.low))

    @property
    def high(self) -> Tensor:
        """Returns the upper bound of the space."""
        return reshape_to_1d_tensor(torch.from_numpy(self._gym_space.high))

    @property
    def shape(self) -> torch.Size:
        """Returns the shape of an element of the space."""
        return self.low.shape

    @staticmethod
    def from_gym(gym_space: gym.Space) -> BoxSpace:
        """Constructs a `BoxSpace` given a Gymnasium `Box` space.

        Args:
            gym_space: A Gymnasium `Box` space.

        Returns:
            A `BoxSpace` with the same bounds and seed as `gym_space`.
        """
        assert isinstance(gym_space, Box)
        return BoxSpace(
            low=torch.from_numpy(gym_space.low),
            high=torch.from_numpy(gym_space.high),
            seed=gym_space._np_random,
        )
