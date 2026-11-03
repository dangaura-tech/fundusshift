"""Verify a mirror against (possibly partial) chunks of the official ZIP release.

Scans the official split-zip chunks for local file headers (PK\\x03\\x04), reads each entry's name, CRC-32 and
uncompressed size, and compares them with the same relative paths in the mirror directory.

python scripts/verify_zip_mirror.py --chunks data/dl/ddr/DDR-dataset.zip.00[1-5] --mirror data/dl/ddr_hf \
    --strip-prefix DDR-dataset/ --only-prefix DR_grading/ --out data/bench/ddr_mirror_verification.csv
"""
import argparse
import struct
import zlib
from pathlib import Path

import pandas as pd

SIG = b"PK\x03\x04"


def local_headers(chunks):
    """Yield (name, crc32, uncompressed_size) for every local header found in the concatenated chunks."""
    data = b"".join(Path(c).read_bytes() for c in chunks)
    pos = data.find(SIG)
    while pos != -1 and pos + 30 <= len(data):
        flags, method, _, _, crc, csize, usize, nlen, xlen = struct.unpack("<HHHHIIIHH", data[pos + 6:pos + 30])
        name = data[pos + 30:pos + 30 + nlen].decode("utf-8", "replace")
        if not flags & 0x08 and not name.endswith("/"):      # sizes/CRC are in the header unless bit 3 is set
            yield name, crc, usize
        pos = data.find(SIG, pos + 30 + nlen + xlen + (csize if not flags & 0x08 else 0))


def crc32_file(p):
    c = 0
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True)
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--strip-prefix", default="")
    ap.add_argument("--only-prefix", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = []
    for name, crc, usize in local_headers(sorted(a.chunks)):
        rel = name[len(a.strip_prefix):] if name.startswith(a.strip_prefix) else name
        if not rel.startswith(a.only_prefix):
            continue
        p = Path(a.mirror) / rel
        if not p.is_file():
            rows.append([rel, "missing_in_mirror"])
        elif p.stat().st_size != usize:
            rows.append([rel, "size_mismatch"])
        else:
            rows.append([rel, "match" if crc32_file(p) == crc else "crc_mismatch"])
    df = pd.DataFrame(rows, columns=["path", "status"])
    df.to_csv(a.out, index=False)
    print(df.status.value_counts().to_string())


if __name__ == "__main__":
    main()
