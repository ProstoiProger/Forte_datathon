import argparse
import sys
import time
from pathlib import Path

import httpx

from cashwatch.contract import validate_result
from cashwatch.video import probe


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a deployed instance against the API contract")
    parser.add_argument("--url", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    headers = {"X-API-Key": args.key}
    duration = probe(args.file).duration_sec
    with httpx.Client(timeout=120.0) as client:
        unauthorized = client.get(f"{base}/api/v1/jobs/0000000000000000")
        print(f"no key -> {unauthorized.status_code}")
        if unauthorized.status_code != 401:
            print("FAIL: requests without a key must be rejected with 401")
            return 1
        started = time.monotonic()
        with args.file.open("rb") as fh:
            response = client.post(f"{base}/api/v1/analyze", headers=headers, files={"file": (args.file.name, fh, "video/quicktime")})
        print(f"upload -> {response.status_code} in {time.monotonic() - started:.1f}s")
        response.raise_for_status()
        job_id = response.json()["job_id"]
        while True:
            body = client.get(f"{base}/api/v1/jobs/{job_id}", headers=headers).json()
            if body["status"] != "processing":
                break
            if time.monotonic() - started > args.timeout:
                print(f"FAIL: no result after {args.timeout:.0f}s")
                return 1
            time.sleep(2.0)
    elapsed = time.monotonic() - started
    if body["status"] == "failed":
        print(f"FAIL: job failed: {body.get('error')}")
        return 1
    errors = validate_result(body, duration=duration)
    print(f"done in {elapsed:.1f}s, verdict={body['verdict']}, violations={len(body['violations'])}")
    for v in body["violations"]:
        print(f"  {v['timecode_sec']:>7.2f}s  {v['type']}  frames={len(v['evidence_frames'])}")
    for error in errors:
        print(f"FAIL: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
