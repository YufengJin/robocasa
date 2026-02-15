#!/usr/bin/env python3
"""
test_random_policy_server.py — Random-action policy gRPC server.

A minimal PolicyService implementation that returns uniformly random actions
sampled from the action spec provided by the client during Reset.
Useful for smoke-testing the gRPC eval pipeline.

Usage:
    python tests/test_random_policy_server.py --port 50051
"""

import argparse
import atexit
import os
import signal
import sys
from concurrent import futures

import grpc
import numpy as np

# ---------------------------------------------------------------------------
# Make sure the workspace root is on sys.path so the generated gRPC stubs
# under robocasa/grpc/ can be imported.
# ---------------------------------------------------------------------------
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from robocasa.grpc import policy_service_pb2, policy_service_pb2_grpc  # noqa: E402

# Global reference so signal handlers can reach it
_server = None


class RandomPolicyServicer(policy_service_pb2_grpc.PolicyServiceServicer):
    """A PolicyService that returns random actions within the action spec."""

    def __init__(self):
        self.action_dim = None
        self.action_low = None
        self.action_high = None
        self.step_count = 0
        self.episode_count = 0

    # ── Reset ───────────────────────────────────────────────────────────
    def Reset(self, request, context):
        self.action_dim = request.action_dim
        self.action_low = np.array(request.action_low, dtype=np.float64)
        self.action_high = np.array(request.action_high, dtype=np.float64)
        self.step_count = 0
        self.episode_count += 1

        print(
            f"[Reset] episode={self.episode_count}  "
            f"task={request.task_name!r}  "
            f"action_dim={self.action_dim}  "
            f"desc={request.task_description!r}"
        )
        return policy_service_pb2.ResetResponse(success=True)

    # ── GetAction ───────────────────────────────────────────────────────
    def GetAction(self, request, context):
        if self.action_low is None or self.action_high is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Reset has not been called yet — action spec unknown.")
            return policy_service_pb2.ActionResponse()

        # Sample a random action within the action spec bounds
        action = np.random.uniform(
            low=self.action_low,
            high=self.action_high,
        )

        self.step_count += 1
        if self.step_count % 100 == 0:
            print(
                f"  [GetAction] step={self.step_count}  "
                f"action[:4]={action[:4].tolist()}"
            )

        return policy_service_pb2.ActionResponse(action=action.tolist())


def _graceful_shutdown(signum=None, frame=None):
    """Stop the gRPC server and release the port. Waits for full shutdown before exit."""
    global _server
    if _server is not None:
        print(f"\nReceived signal {signum}, shutting down server …", flush=True)
        # stop() returns a threading.Event; wait() ensures port is released before we exit
        stop_event = _server.stop(grace=0)
        stop_event.wait(timeout=5)  # grace=0 → stop immediately, wait for port release
        _server = None
        print("Server stopped, port released.", flush=True)
    # Force exit without running atexit/thread join—avoids hang in Docker/Cursor
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def serve(port: int, host: str = "127.0.0.1", max_workers: int = 4):
    """Start the gRPC server."""
    global _server

    options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),    # 50 MB
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50 MB
        ("grpc.so_reuseport", 1),                                # Allow quick re-bind
    ]
    _server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=options,
    )
    policy_service_pb2_grpc.add_PolicyServiceServicer_to_server(
        RandomPolicyServicer(), _server
    )
    bind_addr = f"{host}:{port}"
    _server.add_insecure_port(bind_addr)
    _server.start()
    print(f"Random-action policy server listening on {bind_addr}")
    print("Press Ctrl+C to stop.\n")

    # Register cleanup: Ctrl+C, kill, docker stop, normal exit
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    atexit.register(_graceful_shutdown)

    _server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser(
        description="Random-action policy gRPC server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=50051,
                        help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind (127.0.0.1=local only; 0.0.0.0=all interfaces)")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Max gRPC thread pool workers")
    args = parser.parse_args()
    serve(port=args.port, host=args.host, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
