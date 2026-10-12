import math
import time

PERIOD_S = 120
LOW_MB, HIGH_MB = 100, 600
CHUNK = 16 * 1024 * 1024


def wave(t):
    return 0.5 + 0.5 * math.sin(2 * math.pi * t / PERIOD_S)


buf = bytearray()
t0 = time.time()

while True:
    w = wave(time.time() - t0)

    want = int((LOW_MB + (HIGH_MB - LOW_MB) * w) * 1024 * 1024)
    gap = want - len(buf)
    if gap > 0:
        buf.extend(b"\x01" * min(gap, CHUNK))
    else:
        del buf[want:]

    busy = 0.1 + 0.7 * w
    end = time.time() + busy
    while time.time() < end:
        pass
    time.sleep(1 - busy)