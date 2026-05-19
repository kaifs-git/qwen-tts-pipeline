"""
Performance harness for TTS backends.

Measures per-utterance:
    TTFC      Time-To-First-audio-Chunk (first TTSAudioRawFrame after run_tts call)
    synth_ms  total wall-time from run_tts call to last frame
    audio_s   PCM duration produced (bytes / (sample_rate * 2 * channels))
    RTF       synth_ms / 1000 / audio_s  — real-time factor (<1 = real-time)

Spec targets (from take-home doc):
    TTFC < 60ms
    RTF  < 0.15

Today: runs against `edge` (cloud) and `mock-megakernel` (CPU sine, paced).
Phase C: drop in real `MegakernelTTSService(MegakernelTalkerBackend(...))` and
this same harness reports the assignment-grade numbers.

Usage:
    venv/bin/python -m bench.perf                              # all backends, default prompts
    venv/bin/python -m bench.perf --backends mock-megakernel   # one backend
    venv/bin/python -m bench.perf --runs 10 --warmup 3
    venv/bin/python -m bench.perf --output bench/results/run1.json
    venv/bin/python -m bench.perf --no-pace                    # mock w/ pace_rtf=0 (raw throughput)
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.talker_backend import MockTalkerBackend  # noqa: E402  (after sys.path bootstrap)
from pipeline.tts_edge import EdgeTTSService  # noqa: E402
from pipeline.tts_megakernel import MegakernelTTSService  # noqa: E402

# fmt: off
PROMPTS = {
    "short":  "The quick brown fox jumps over the lazy dog.",
    "medium": "Speech synthesis benchmarks should measure first-chunk latency and real-time factor, not just throughput. A fast model with high startup latency feels slow.",
    "long":   "Real-time voice agents live and die by latency. The user perceives lag from the moment they stop speaking until the first sound comes back. Time to first audio chunk, often called TTFC or first-byte latency, captures that window. Real-time factor, the ratio of synthesis wall-time to audio duration, decides whether the agent can keep up over a long response. A megakernel that hits one thousand tokens per second on paper is irrelevant if the codec stage adds half a second of buffering on the way out.",
}
# fmt: on

DEFAULT_BACKENDS = ("edge", "mock-megakernel")
SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
CHANNELS = 1


@dataclass
class RunResult:
    backend: str
    prompt_key: str
    prompt_words: int
    ttfc_ms: float
    synth_ms: float
    audio_s: float
    rtf: float
    chunks: int
    error: Optional[str] = None


@dataclass
class Aggregate:
    backend: str
    prompt_key: str
    n: int
    ttfc_ms_p50: float
    ttfc_ms_p95: float
    synth_ms_p50: float
    audio_s_mean: float
    rtf_p50: float
    rtf_p95: float
    runs: list = field(default_factory=list)


def build_service(name: str, pace_rtf: float):
    if name == "edge":
        return EdgeTTSService(voice="en-US-AriaNeural")
    if name == "mock-megakernel":
        return MegakernelTTSService(backend=MockTalkerBackend(pace_rtf=pace_rtf))
    raise ValueError(f"unknown backend: {name}")


async def run_once(service, prompt: str, ctx: str) -> RunResult:
    t0 = time.perf_counter()
    ttfc_ms = None
    total_bytes = 0
    chunks = 0
    err = None
    try:
        async for f in service.run_tts(prompt, ctx):
            cls = type(f).__name__
            if cls == "TTSAudioRawFrame":
                if ttfc_ms is None:
                    ttfc_ms = (time.perf_counter() - t0) * 1000
                chunks += 1
                total_bytes += len(f.audio)
            elif cls == "ErrorFrame":
                err = getattr(f, "error", str(f))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    synth_ms = (time.perf_counter() - t0) * 1000
    audio_s = total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS)
    rtf = (synth_ms / 1000.0) / audio_s if audio_s > 0 else float("nan")
    return RunResult(
        backend=type(service).__name__,
        prompt_key="",
        prompt_words=len(prompt.split()),
        ttfc_ms=ttfc_ms if ttfc_ms is not None else float("nan"),
        synth_ms=synth_ms,
        audio_s=audio_s,
        rtf=rtf,
        chunks=chunks,
        error=err,
    )


async def bench_backend(name: str, runs: int, warmup: int, pace_rtf: float):
    service = build_service(name, pace_rtf)
    results = []
    for prompt_key, prompt in PROMPTS.items():
        # warmup
        for w in range(warmup):
            await run_once(service, prompt, f"warmup-{name}-{prompt_key}-{w}")
        # measured
        for r in range(runs):
            res = await run_once(service, prompt, f"run-{name}-{prompt_key}-{r}")
            res.backend = name
            res.prompt_key = prompt_key
            results.append(res)
            print(
                f"  {name:<18} {prompt_key:<7} "
                f"TTFC={res.ttfc_ms:6.0f}ms  "
                f"synth={res.synth_ms:6.0f}ms  "
                f"audio={res.audio_s:5.2f}s  "
                f"RTF={res.rtf:5.2f}  "
                f"chunks={res.chunks}"
                + (f"  ERR={res.error}" if res.error else "")
            )
    return results


def aggregate(results: list[RunResult]) -> list[Aggregate]:
    groups: dict[tuple, list[RunResult]] = {}
    for r in results:
        groups.setdefault((r.backend, r.prompt_key), []).append(r)
    aggs = []
    for (backend, key), rs in groups.items():
        ok = [r for r in rs if not r.error and r.audio_s > 0]
        if not ok:
            continue
        ttfcs = sorted(r.ttfc_ms for r in ok)
        synths = sorted(r.synth_ms for r in ok)
        rtfs = sorted(r.rtf for r in ok)
        aggs.append(
            Aggregate(
                backend=backend,
                prompt_key=key,
                n=len(ok),
                ttfc_ms_p50=statistics.median(ttfcs),
                ttfc_ms_p95=ttfcs[max(0, int(len(ttfcs) * 0.95) - 1)],
                synth_ms_p50=statistics.median(synths),
                audio_s_mean=statistics.mean(r.audio_s for r in ok),
                rtf_p50=statistics.median(rtfs),
                rtf_p95=rtfs[max(0, int(len(rtfs) * 0.95) - 1)],
                runs=[asdict(r) for r in ok],
            )
        )
    return aggs


def print_table(aggs: list[Aggregate], targets=(60.0, 0.15)):
    ttfc_target, rtf_target = targets
    print()
    print("=" * 96)
    print(
        f"{'backend':<18} {'prompt':<7} {'n':>3} "
        f"{'TTFC p50':>10} {'TTFC p95':>10} {'synth p50':>11} "
        f"{'audio s':>9} {'RTF p50':>9} {'RTF p95':>9}"
    )
    print("-" * 96)
    for a in aggs:
        ttfc_mark = "✓" if a.ttfc_ms_p50 < ttfc_target else "✗"
        rtf_mark = "✓" if a.rtf_p50 < rtf_target else "✗"
        print(
            f"{a.backend:<18} {a.prompt_key:<7} {a.n:>3} "
            f"{a.ttfc_ms_p50:>9.0f}{ttfc_mark} {a.ttfc_ms_p95:>10.0f} "
            f"{a.synth_ms_p50:>11.0f} {a.audio_s_mean:>9.2f} "
            f"{a.rtf_p50:>8.2f}{rtf_mark} {a.rtf_p95:>9.2f}"
        )
    print("=" * 96)
    print(
        f"targets: TTFC p50 < {ttfc_target:.0f}ms  (✓/✗)   "
        f"RTF p50 < {rtf_target:.2f}  (✓/✗)"
    )
    print(
        "note: edge-tts numbers reflect cloud TTS round-trip, NOT the megakernel "
        "spec — those land in Phase C on vast.ai."
    )


async def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--backends",
        nargs="+",
        choices=DEFAULT_BACKENDS,
        default=list(DEFAULT_BACKENDS),
    )
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--no-pace",
        action="store_true",
        help="mock backend: disable RTF pacing (raw throughput)",
    )
    p.add_argument(
        "--output", type=str, default=None, help="write full JSON results here"
    )
    args = p.parse_args()

    pace_rtf = 0.0 if args.no_pace else 0.15
    all_results: list[RunResult] = []
    for b in args.backends:
        print(f"\n--- {b} ---")
        all_results.extend(await bench_backend(b, args.runs, args.warmup, pace_rtf))

    aggs = aggregate(all_results)
    print_table(aggs)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(a) for a in aggs], indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
