"""Provenance manifest for the official ROADEF 2016 instance files.

``roadef_2016_data/`` is gitignored, so the repository itself cannot attest
that the instance XMLs on disk are the ones published at
https://roadef.org/challenge/2016/en/instances.php.  This module pins the
SHA-256 of every Set B and Set X instance as downloaded from the official
site (byte-compared 2026-08-19) and classifies a file against them.

Classification is informational, not a gate: solving a simulated or private
instance is a supported workflow.  The one loud case is a file whose *name*
matches an official instance but whose bytes do not -- that is a modified
benchmark and any result on it must not be reported as official.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# SHA-256 of each instance XML inside the official archives
# Instances_B_V25-11042016.zip and "Instances X.zip".
OFFICIAL_INSTANCE_SHA256: dict[str, str] = {
    "V2.12.xml": "251f8d62be68d06c7b96564d800787cf9eeaae03bb31c6415f19492f780f4f09",
    "V2.13.xml": "c18f11d30bb0fd24c80df72169952b8c6a2b9fff6407add9e2ec0bc1bc5a0002",
    "V2.14.xml": "9a25bd462b6be72369febf3a2cbdd04cd817836c5d2997514bee3116ed702106",
    "V2.15.xml": "0bf02e6e34f7fb86566f23feec14937ae1dba44aa1268235dd35583b9d553e72",
    "V2.16.2.xml": "266030a1179b11d20831f585fe3e24af4edd631758062fd9b7970de975c43e46",
    "V2.17.xml": "2016770039bedb870dfc0d50f66b545049fb5eed891df8f4832a24b2d46ba9bd",
    "V2.18.xml": "fb95d91cb391339cac7ab94152c7c5c7afa5766c762409659fd5585caa7345c9",
    "V2.19.xml": "b0db14c8e5aee69755b802f17096a67d043a488ce22711af8197a6000339811c",
    "V2.20.2.xml": "180f9cd0850c71d17718b02a03fa5e3e0877c4139870a9840b9860e276b64c94",
    "V2.21.2.xml": "e88c72fb8e44d72dee501f4e4343ba297e14d0385389f1633d0e60bdfac31cbd",
    "V2.22.xml": "413f4bee62fb99884871a3ee7cc5cf60751fff3eb2523bb0f22bcb0134becdcc",
    "V2.23.xml": "cc84587182a5ff734d8b8b8e9283eabe4cb5d9b62c8bd70419b437b431a82b00",
    "V2.24.xml": "f6e89781b2897b3c35ecd5a71bf0c8b679ce6ed41488f45dbc8bf946b3f729e6",
    "V2.25.xml": "b92664bda70e9f71ce60f87fa00886586ea1a224c46b46de6dbfea5825c92077",
    "V2.26.xml": "232761c891d1abb3bc13998b5dad6ed61da9f29bbab5254da55d1ea4501b9c48",
    "X1.xml": "1b25c675df7f99973b2b6a63a63102bf09d13ef504c209e0702250999a8466ba",
    "X2.xml": "9bd4217330019f5ff65085c47d59ef94c9f32935cf07d0be2fded9da31f01e10",
    "X3.xml": "8797703cc720c1e721c0962862f1271eeea0b02537e40cfe4a0cddb78c6de314",
    "X4.xml": "56cae25c0cbed3ff5f178ba2baefaadcf81d5fccd6934423faa29d9846938760",
    "X5.xml": "bf7af1446c1b61bbe3699e966fa96ec2a2add2b14ceefb557f5873cbe21d47ac",
}


def classify_instance(path: Path | str) -> str:
    """Classify an instance file against the official manifest.

    Returns one of:

    - ``"official"`` -- name and bytes match the roadef.org download.
    - ``"MODIFIED-OFFICIAL"`` -- the filename is an official instance's but
      the content differs.  Results on this file are not benchmark results.
    - ``"not-in-manifest"`` -- a private or simulated instance.
    """
    file = Path(path)
    expected = OFFICIAL_INSTANCE_SHA256.get(file.name)
    if expected is None:
        return "not-in-manifest"
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "official" if digest.hexdigest() == expected else "MODIFIED-OFFICIAL"
