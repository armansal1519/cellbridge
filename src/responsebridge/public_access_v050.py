"""Public-cohort access checks: size, URL, metadata-only sample tables.

Never opens biological ADT values for sealed cohorts.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen
import json
import re

from .artifacts import write_json


def head_url(url, timeout=30):
    request = Request(url, method="HEAD")
    request.add_header("User-Agent", "ResponseBridge/0.5 (research; public-data access check)")
    with urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        return {"url": url, "status": int(response.status),
                "content_length": None if length is None else int(length),
                "content_type": response.headers.get("Content-Type"),
                "final_url": response.geturl()}


def fetch_text(url, timeout=60, max_bytes=4_000_000):
    request = Request(url)
    request.add_header("User-Agent", "ResponseBridge/0.5 (research; metadata-only)")
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(max_bytes)
        return payload.decode("utf-8", errors="replace"), int(response.status)


def gse_sample_accessions(soft_text):
    samples = re.findall(r"^!Sample_geo_accession = (GSM\d+)", soft_text, flags=re.M)
    titles = re.findall(r"^!Sample_title = (.+)$", soft_text, flags=re.M)
    return {"n_sample_accessions": len(samples), "sample_accessions": samples,
            "n_titles": len(titles), "titles_head": titles[:40]}


def write_access_report(path, rows):
    path = Path(path)
    write_json(path, {"version": "0.5.0", "rows": rows}, immutable=True)
    return path
