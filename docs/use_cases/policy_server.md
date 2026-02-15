# Policy to PolicyServer Guide

This guide explains how to wrap any policy object into a gRPC PolicyService server for evaluation. Reference implementations: `robocasa/grpc/`, `tests/test_random_policy_server.py`, and `scripts/run_eval.py`.

## 1. gRPC Interface Contract

- **Reset**: Called at the start of each episode. Receives `task_name`, `task_description`, `action_dim`, `action_low`, `action_high`.
- **GetAction**: Called every timestep. Receives `ObservationRequest` and returns `ActionResponse`.

## 2. Required Servicer Implementation

Implement a class that inherits `policy_service_pb2_grpc.PolicyServiceServicer` and implements `Reset` and `GetAction`:

```python
from robocasa.grpc import policy_service_pb2, policy_service_pb2_grpc

class MyPolicyServicer(policy_service_pb2_grpc.PolicyServiceServicer):
    def __init__(self):
        self.policy = ...  # Your policy object

    def Reset(self, request, context):
        # Parse request.action_dim, request.action_low, request.action_high
        # Call policy.reset() or equivalent initialization
        return policy_service_pb2.ResetResponse(success=True)

    def GetAction(self, request, context):
        # Parse ObservationRequest → obs dict
        # Call policy.get_action(obs) or policy(obs)
        # Return policy_service_pb2.ActionResponse(action=action.tolist())
        ...
```

## 3. Policy Object Adaptation

| Policy Native Interface | Reset Adapter | GetAction Adapter |
|------------------------|---------------|-------------------|
| `policy.reset(task_name, task_desc, action_spec)` | Extract from `request` and call | — |
| `policy(obs)` / `policy.get_action(obs)` | — | Build `obs` from `request`, call and return |
| Image format | — | `request.primary_image` etc. are JPEG bytes; decode to `np.ndarray` (H,W,3) RGB |
| Observation keys | — | Client convention: `primary_image`, `secondary_image`, `wrist_image`, `proprio` |

## 4. Parsing ObservationRequest in GetAction

```python
import cv2
import numpy as np

def _decode_jpeg(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def GetAction(self, request, context):
    obs = {
        "primary_image": _decode_jpeg(request.primary_image),
        "secondary_image": _decode_jpeg(request.secondary_image),
        "wrist_image": _decode_jpeg(request.wrist_image),
        "proprio": np.array(request.proprio, dtype=np.float64),
        "task_description": request.task_description,
    }
    action = self.policy(obs)  # or self.policy.get_action(obs)
    return policy_service_pb2.ActionResponse(action=action.tolist())
```

## 5. Action Dimension and Client Convention

- The eval client `run_eval.py` supports 7-dim policy output: it will automatically pad to 12-dim (mobile base, etc.).
- Returned `action` must be `list[float]`, with length equal to `action_dim` or 7 (client will pad).

## 6. Starting the gRPC Server

```python
policy_service_pb2_grpc.add_PolicyServiceServicer_to_server(
    MyPolicyServicer(), server
)
```

Use the same options as `test_random_policy_server.py`: 50MB message limit, SO_REUSEPORT, and graceful shutdown (SIGINT/SIGTERM/atexit).

## 7. Testing

1. Start the policy server: `python <your_policy_server.py> --port 50051`
2. Run the eval client: `python scripts/run_eval.py --policy_server_addr localhost:50051 ...`
