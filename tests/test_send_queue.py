"""The bounded queue of in-flight SWIFT broadcasts.

Driven with stub requests rather than MPI: what matters is when the queue
blocks, and a real small message completes eagerly, which is precisely the case
that never blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from swift.comm.swift import _SendQueue


class FakeRequest:
    """A send that lands only when told to."""

    def __init__(self) -> None:
        self.landed = False
        self.waited = False

    def Test(self) -> bool:
        return self.landed

    def Wait(self) -> None:
        self.waited = True
        self.landed = True


def _payload() -> np.ndarray:
    return np.zeros(4, dtype=np.float32)


def test_landed_sends_are_retired() -> None:
    queue = _SendQueue(capacity=8)
    requests = [FakeRequest() for _ in range(3)]
    for request in requests:
        queue.add(request, _payload())
    assert len(queue._queue) == 3

    for request in requests:
        request.landed = True
    queue.reap()
    assert len(queue._queue) == 0


def test_a_full_queue_blocks_on_the_oldest_send() -> None:
    """Backpressure for a neighbour that has fallen behind."""
    queue = _SendQueue(capacity=2)
    requests = [FakeRequest() for _ in range(4)]
    for request in requests:
        queue.add(request, _payload())

    assert requests[0].waited, "the oldest send should have been waited on"
    assert len(queue._queue) <= 3


def test_full_reports_capacity_without_blocking() -> None:
    queue = _SendQueue(capacity=2)
    held = [FakeRequest() for _ in range(3)]
    for request in held:
        queue.add(request, _payload(), block=False)

    assert queue.full
    assert not any(request.waited for request in held), "nothing should have blocked"


def test_non_blocking_add_never_waits() -> None:
    """finish() uses this: waiting there would stop it reaching the test that
    ends the shutdown loop."""
    queue = _SendQueue(capacity=1)
    requests = [FakeRequest() for _ in range(20)]
    for request in requests:
        queue.add(request, _payload(), block=False)
    assert not any(request.waited for request in requests)


def test_payload_is_retained_until_the_send_lands() -> None:
    """MPI reads the buffer asynchronously, so the queue must keep it alive."""
    queue = _SendQueue(capacity=4)
    request = FakeRequest()
    payload = _payload()
    queue.add(request, payload)
    assert queue._queue[0][1] is payload

    request.landed = True
    queue.reap()
    assert not queue._queue


def test_wait_all_is_a_no_op_when_empty() -> None:
    _SendQueue(capacity=2).wait_all()


def test_capacity_zero_still_accepts_a_non_blocking_add() -> None:
    queue = _SendQueue(capacity=0)
    request = FakeRequest()
    queue.add(request, _payload(), block=False)
    assert not request.waited
    assert queue.full


@pytest.mark.parametrize("capacity", [1, 2, 16])
def test_blocking_add_keeps_the_queue_bounded(capacity: int) -> None:
    queue = _SendQueue(capacity=capacity)
    for _ in range(50):
        queue.add(FakeRequest(), _payload())
    assert len(queue._queue) <= capacity + 1
