#!/usr/bin/env python3
"""
Test policy server using ActionChunkBroker — predicts 16-step chunks, executes 8 steps.

Demonstrates "predict 16, execute 8" : the inner policy returns (16, action_dim)
chunks; ActionChunkBroker yields one action per infer and re-queries every 8 steps.

Usage:
    python tests/test_ac_policy_server.py --port 8000

Then connect with:
    python scripts/run_demo.py --policy_server_addr localhost:8000 --task_name PnPCounterToCab
    python scripts/run_eval.py --policy_server_addr localhost:8000 --task_name PnPCounterToCab
"""

import argparse
import logging
import sys
from typing import Dict

import numpy as np

from policy_websocket import BasePolicy, WebsocketPolicyServer, ActionChunkBroker

logger = logging.getLogger(__name__)

CHUNK_SIZE = 16
EXECUTE_STEPS = 8


class RandomChunkPolicy(BasePolicy):
    """Returns random action chunks of shape (chunk_size, action_dim)."""

    def __init__(self, chunk_size: int = CHUNK_SIZE):
        self._chunk_size = chunk_size
        self._action_dim: int = 7
        self._action_low: np.ndarray = np.full(7, -1.0)
        self._action_high: np.ndarray = np.full(7, 1.0)
        self._scale: float = 0.1

    def infer(self, obs: Dict) -> Dict:
        if "action_dim" in obs:
            self._action_dim = int(obs["action_dim"])
            self._action_low = np.array(obs.get("action_low", np.full(self._action_dim, -1.0)))
            self._action_high = np.array(obs.get("action_high", np.full(self._action_dim, 1.0)))

        low = self._action_low * self._scale
        high = self._action_high * self._scale
        actions = np.random.uniform(low, high, (self._chunk_size, self._action_dim)).astype(
            np.float64
        )
        return {"actions": actions}

    def reset(self) -> None:
        pass


class ResetOnInitPolicy(BasePolicy):
    """Calls reset() when receiving episode-init obs (action_dim only, no images)."""

    def __init__(self, policy: BasePolicy):
        self._policy = policy

    def infer(self, obs: Dict) -> Dict:
        if "action_dim" in obs and "primary_image" not in obs:
            self._policy.reset()
        return self._policy.infer(obs)

    def reset(self) -> None:
        self._policy.reset()


def main():
    parser = argparse.ArgumentParser(
        description="RoboCasa test policy server (action chunks: predict 16, execute 8)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--execute_steps", type=int, default=EXECUTE_STEPS)
    args = parser.parse_args()

    chunk_policy = RandomChunkPolicy(chunk_size=args.chunk_size)
    broker = ActionChunkBroker(
        policy=chunk_policy,
        action_horizon=args.execute_steps,
    )
    policy = ResetOnInitPolicy(broker)

    metadata = {
        "policy_name": "RandomChunkPolicy+ActionChunkBroker",
        "action_dim": 7,
        "chunk_size": args.chunk_size,
        "execute_steps": args.execute_steps,
    }

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    print(
        f"Starting ActionChunk policy server on ws://{args.host}:{args.port} "
        f"(predict {args.chunk_size}, execute {args.execute_steps})"
    )
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as e:
        if e.errno == 98:
            print(f"\nERROR: Port {args.port} is already in use.")
            print(f"Kill with: lsof -ti :{args.port} | xargs kill -9")
            sys.exit(1)
        raise
    print("Server stopped, port released.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
