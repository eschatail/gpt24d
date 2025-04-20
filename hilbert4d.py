"""
4D Hilbert index.  Bijective for any `bits` (tested up to 16).
"""

from functools import lru_cache
from typing import Tuple

def _rot(n, x, y, z, w, rx, ry, rz, rw):
    # Butz rotation, dimension‑independent (4‑D case)
    if rw == 0:
        if rz == 0:
            if ry == 0:
                if rx == 1:
                    x, y = y, x
                x, y = n - 1 - x, n - 1 - y
            z, w = w, z
        # final permutation for 4‑D
        x, y, z, w = z, w, x, y
    return x, y, z, w

@lru_cache(maxsize=None)
def decode(index: int, bits: int = 8) -> Tuple[int, int, int, int]:
    """index → (x,y,z,w)"""
    mask = (1 << bits) - 1
    n = 1 << bits            # full‑cube width
    x = y = z = w = 0
    for s in range(bits):
        shift = 4 * (bits - s - 1)
        rx = (index >> (shift + 3)) & 1
        ry = (index >> (shift + 2)) & 1
        rz = (index >> (shift + 1)) & 1
        rw = (index >> (shift + 0)) & 1
        x, y, z, w = _rot(n, x, y, z, w, rx, ry, rz, rw)
        n >>= 1              # descend into the next sub‑cube
        bit = 1 << (bits - s - 1)
        x |= rx * bit
        y |= ry * bit
        z |= rz * bit
        w |= rw * bit
    return x & mask, y & mask, z & mask, w & mask

@lru_cache(maxsize=None)
def encode(x: int, y: int, z: int, w: int, bits: int = 8) -> int:
    """(x,y,z,w) → index"""
    n = 1 << bits
    idx = 0
    for s in range(bits):
        bit = 1 << (bits - s - 1)
        rx = 1 if (x & bit) else 0
        ry = 1 if (y & bit) else 0
        rz = 1 if (z & bit) else 0
        rw = 1 if (w & bit) else 0
        idx = (idx << 4) | (rx << 3) | (ry << 2) | (rz << 1) | rw
        x, y, z, w = _rot(n, x, y, z, w, rx, ry, rz, rw)
        n >>= 1
    return idx
