import math
import random
import time

PERIOD_S = 120
SLOW_PERIOD_S = 900
NOISE = 0.15
LOW_MB, HIGH_MB = 100, 600
PAGE = b"\x01" * (16 * 1024 * 1024)


def level(t):
    fast = 0.5 + 0.5 * math.sin(2 * math.pi * t / PERIOD_S)
    slow = 0.5 + 0.5 * math.sin(2 * math.pi * t / SLOW_PERIOD_S)
    raw = 0.7 * fast + 0.3 * slow + random.gauss(0, NOISE)
    return min(1.0, max(0.0, raw))


buf = bytearray()
t0 = time.time()

while True:
    w = level(time.time() - t0)

    want = int((LOW_MB + (HIGH_MB - LOW_MB) * w) * 1024 * 1024)
    while len(buf) + len(PAGE) <= want:
        buf.extend(PAGE)
    if len(buf) > want:
        del buf[want:]

    busy = 0.1 + 0.7 * w
    end = time.time() + busy
    while time.time() < end:
        pass
    time.sleep(1 - busy)