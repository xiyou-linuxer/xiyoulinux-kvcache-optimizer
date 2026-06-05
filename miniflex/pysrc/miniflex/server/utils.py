import os
from typing import Optional

import zmq

DEFAULT_ZMQ_HWM = 64
DEFAULT_ZMQ_LINGER_MS = 0


def normalize_ipc_endpoint(path_or_endpoint: str) -> str:
  if not isinstance(path_or_endpoint, str) or path_or_endpoint == "":
    raise ValueError("path_or_endpoint must be a non-empty string")
  if path_or_endpoint.startswith(("ipc://", "tcp://")):
    return path_or_endpoint
  return f"ipc://{path_or_endpoint}"


def make_tcp_endpoint(host: str, port: int | str) -> str:
  if not host:
    raise ValueError("host must be a non-empty string")
  port_int = int(port)
  if port_int <= 0 or port_int > 65535:
    raise ValueError(f"port must be in [1, 65535], got {port}")
  return f"tcp://{host}:{port_int}"


def normalize_tcp_endpoint(endpoint_or_port: str | int, host: str = "127.0.0.1") -> str:
  if isinstance(endpoint_or_port, int):
    return make_tcp_endpoint(host, endpoint_or_port)
  if endpoint_or_port.startswith("tcp://"):
    return endpoint_or_port
  if endpoint_or_port.isdigit():
    return make_tcp_endpoint(host, int(endpoint_or_port))
  raise ValueError(
    "endpoint_or_port must be a tcp:// endpoint, integer port, or numeric port string"
  )


def normalize_zmq_endpoint(path_or_endpoint: str) -> str:
  return normalize_ipc_endpoint(path_or_endpoint)


def get_zmq_socket(
    context: zmq.Context,
    socket_type: zmq.SocketType,
    endpoint: str,
    bind: bool = True,
    *,
    hwm: int = DEFAULT_ZMQ_HWM,
    linger_ms: int = DEFAULT_ZMQ_LINGER_MS,
    recv_timeout_ms: Optional[int] = None,
    send_timeout_ms: Optional[int] = None,
) -> zmq.Socket:
  if socket_type not in (zmq.PUSH, zmq.PULL, zmq.DEALER):
    raise ValueError(f"unsupported ZMQ socket type: {socket_type}")
  if hwm < 0:
    raise ValueError(f"hwm must be non-negative, got {hwm}")

  endpoint = normalize_zmq_endpoint(endpoint)
  socket = context.socket(socket_type)
  socket.setsockopt(zmq.LINGER, linger_ms)

  if socket_type in (zmq.PUSH, zmq.DEALER):
    socket.setsockopt(zmq.SNDHWM, hwm)
  if socket_type in (zmq.PULL, zmq.DEALER):
    socket.setsockopt(zmq.RCVHWM, hwm)
  if recv_timeout_ms is not None:
    socket.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
  if send_timeout_ms is not None:
    socket.setsockopt(zmq.SNDTIMEO, send_timeout_ms)

  if bind:
    if endpoint.startswith("ipc://"):
      ipc_path = endpoint[len("ipc://"):]
      try:
        os.unlink(ipc_path)
      except FileNotFoundError:
        pass
    socket.bind(endpoint)
  else:
    socket.connect(endpoint)
  return socket
