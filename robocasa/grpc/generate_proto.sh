#!/usr/bin/env bash
# Generate Python gRPC stubs from policy_service.proto
# Usage: cd robocasa/grpc && bash generate_proto.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m grpc_tools.protoc \
    -I"${SCRIPT_DIR}" \
    --python_out="${SCRIPT_DIR}" \
    --grpc_python_out="${SCRIPT_DIR}" \
    "${SCRIPT_DIR}/policy_service.proto"

# Fix the import in the generated gRPC stub so it works as a package import
# (grpc_tools generates "import policy_service_pb2" which only works if the
# directory itself is on sys.path).
sed -i 's/^import policy_service_pb2 as/from robocasa.grpc import policy_service_pb2 as/' \
    "${SCRIPT_DIR}/policy_service_pb2_grpc.py"

echo "Proto stubs generated in ${SCRIPT_DIR}"
