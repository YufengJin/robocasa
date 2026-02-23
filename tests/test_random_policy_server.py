#!/usr/bin/env python3
"""
Test policy server for RoboCasa — returns random actions via WebSocket.

Usage:
    python tests/test_random_policy_server.py --port 8000

Then connect with:
    python scripts/run_demo.py --policy_server_addr localhost:8000 --task_name PnPCounterToCab
"""

import argparse
import logging
from typing import Dict

import numpy as np

from policy_websocket import BasePolicy, WebsocketPolicyServer

logger = logging.getLogger(__name__)


class RandomPolicy(BasePolicy):
    """Returns uniformly random actions within the given action spec."""

    def __init__(self) -> None:
        self._action_dim: int = 7
        self._action_low: np.ndarray = np.full(7, -1.0)
        self._action_high: np.ndarray = np.full(7, 1.0)
        self._scale: float = 0.1

    def infer(self, obs: Dict) -> Dict:
        if "action_dim" in obs:
            self._action_dim = int(obs["action_dim"])
            self._action_low = np.array(obs.get("action_low", np.full(self._action_dim, -1.0)))
            self._action_high = np.array(obs.get("action_high", np.full(self._action_dim, 1.0)))

        action = np.random.uniform(
            self._action_low * self._scale,
            self._action_high * self._scale,
        ).astype(np.float64)

        return {"actions": action}

    def reset(self) -> None:
        pass


def main():
    parser = argparse.ArgumentParser(description="RoboCasa test policy server (random actions)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    policy = RandomPolicy()
    metadata = {"policy_name": "RandomPolicy", "action_dim": 7}

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    print(f"Starting RandomPolicy server on ws://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Server stopped, port released.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
