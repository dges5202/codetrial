#!/usr/bin/env python3
"""Break down CodeTrial reply latency from the agent's server log.

Reads `codetrial latency {JSON}` records, joins each reply's records by
(pid, room, session, turn), and partitions origin-to-first-native-capture into
consecutive segments so the dominant one is visible per turn and in aggregate.
"""

import argparse
import json
import math
import sys
from collections import Counter

MARKER = "codetrial latency "
VERSION = 2
ORIGINS = {
    "last_voiced_pcm",
    "prompt",
    "transcript",
    "provider_voice_end",
    "provider_event",
}
STAGES = {
    "tool_response",
    "audio_accepted",
    "native_audio_capture",
    "native_audio_cancelled",
    "audio_enqueue_failed",
    "turn_complete",
    "interrupted",
    "paused",
    "held",
    "socket_replaced",
}
TIMINGS = (
    "elapsed_ms",
    "observed_ms",
    "provider_voice_end_ms",
    "first_transcript_ms",
    "latest_transcript_ms",
    "candidate_yield_ms",
    "first_provider_output_ms",
    "first_tool_call_ms",
    "tool_ack_ms",
    "last_tool_response_ms",
    "tool_handler_ms",
    "provider_event_queue_ms",
    "stale_drop_ms",
    "enqueue_ms",
    "handler_tail_ms",
    "playout_ahead_ms",
    "output_queue_ms",
    "native_capture_ms",
    "server_total_ms",
    "tool_batch_ms",
    "tool_execute_ms",
    "board_resend_ms",
    "tool_send_ms",
    "silence_window_ms",
)
# Consecutive, so they sum to the server total. The configured silence window
# is a comparison allowance, not a measured provider span: explicit yielding
# and overlapping inference can change the actual endpointing wait.
SEGMENTS = (
    "candidate_yield",
    "silence_window",
    "endpointing",
    "response_start",
    "tool_decision",
    "tool_rounds",
    "post_tool_response",
    "output_to_audio",
    "output_queue",
    "native_capture",
)
SEGMENT_HELP = {
    "candidate_yield": "candidate's own time before choosing their turn is done",
    "silence_window": "configured silence allowance, not measured provider time",
    "endpointing": "rest of origin to provider END (uplink, VAD, any work before END)",
    "response_start": "END (or origin) to first provider output",
    "tool_decision": "END (or origin) to first tool call",
    "tool_rounds": "first tool call to last tool response sent",
    "post_tool_response": "last tool response to first audio accepted",
    "output_to_audio": "first output (transcript) to first audio accepted",
    "output_queue": "audio accepted to worker dequeue",
    "native_capture": "LiveKit capture_frame of the first frame",
}


class Rejected(ValueError):
    pass


def is_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    # Production timestamps fit i64 and durations fit u64. Bound parsed
    # numbers too, so malformed logs cannot overflow aggregation arithmetic.
    return -(2**63) <= value <= 2**64 - 1 and math.isfinite(value)


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def parse(line):
    """The record on this line, None for a line without one, or Rejected."""
    if MARKER not in line:
        return None
    payload = line.split(MARKER, 1)[1].strip()
    try:
        record = json.loads(payload)
    except (ValueError, RecursionError) as error:
        raise Rejected("not JSON") from error
    if not isinstance(record, dict):
        raise Rejected("not an object")
    if record.get("event") != "reply_latency" or record.get("version") != VERSION:
        raise Rejected("unsupported event or version")
    if not isinstance(record.get("room"), str):
        raise Rejected("room")
    for field in ("pid", "session", "socket", "turn"):
        if not is_int(record.get(field)):
            raise Rejected(field)
    if not isinstance(record.get("stage"), str) or record["stage"] not in STAGES:
        raise Rejected("stage")
    if not isinstance(record.get("origin"), str) or record["origin"] not in ORIGINS:
        raise Rejected("origin")
    for field in TIMINGS:
        value = record.get(field)
        if value is not None and not is_number(value):
            raise Rejected(field)
    signed_milestones = {
        "observed_ms",
        "provider_voice_end_ms",
        "first_transcript_ms",
        "latest_transcript_ms",
        "candidate_yield_ms",
        "first_provider_output_ms",
        "first_tool_call_ms",
        "last_tool_response_ms",
    }
    for field in TIMINGS:
        if field not in signed_milestones and (record.get(field) or 0) < 0:
            raise Rejected(field)
    for field in (
        "tool_batches",
        "tool_count",
        "origin_socket",
        "output_queue_frames",
        "context_tokens",
    ):
        if record.get(field) is not None and (
            not is_int(record[field]) or record[field] < 0
        ):
            raise Rejected(field)
    names = record.get("tool_names")
    if names is not None and not (
        isinstance(names, list) and all(isinstance(n, str) for n in names)
    ):
        raise Rejected("tool_names")
    for field in ("captured", "sent", "enqueued", "continuing"):
        if field in record and not isinstance(record[field], bool):
            raise Rejected(field)
    if record.get("captured") is True and record["stage"] != "native_audio_capture":
        raise Rejected("capture stage")
    if record["stage"] == "native_audio_capture" and record.get("captured") is True:
        if record.get("elapsed_ms") is None or record.get("server_total_ms") is None:
            raise Rejected("missing capture timing")
    return record


def offset(record):
    value = record.get("server_total_ms")
    return value if value is not None else record.get("elapsed_ms")


# Below this many values a p95 is one of the last two, so a single reply moves
# it; the summary marks such statistics instead of hiding them.
SMALL_SAMPLE = 20


def percentile(values, fraction):
    """Linear interpolation between closest ranks, as numpy's default does."""
    rank = (len(values) - 1) * fraction
    low = math.floor(rank)
    high = min(low + 1, len(values) - 1)
    value = round(values[low] + (values[high] - values[low]) * (rank - low), 1)
    return int(value) if value == int(value) else value


def statistics(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min_ms": values[0],
        "p50_ms": percentile(values, 0.50),
        "mean_ms": round(sum(values) / len(values), 1),
        "p95_ms": percentile(values, 0.95),
        "max_ms": values[-1],
        "small_sample": len(values) < SMALL_SAMPLE,
    }


# Context sizes a reply started from, in tokens: each band's upper bound.
CONTEXT_BANDS = (8_000, 16_000, 32_000, 64_000)


def context_band(tokens):
    lower = 0
    for upper in CONTEXT_BANDS:
        if tokens < upper:
            return f"{lower // 1000}k_{upper // 1000}k"
        lower = upper
    return f"{lower // 1000}k_up"


def slope_per_1k(points):
    """Least-squares ms per thousand tokens within each kind of reply, or None
    without spread.

    Points are (kind, tokens, ms). Each kind is centered on its own means
    first: tool replies cost more and come later in an interview, so a slope
    fitted across kinds reads that mix as growth with context.
    """
    by_kind = {}
    for kind, x, y in points:
        by_kind.setdefault(kind, []).append((x, y))
    centered = []
    for pairs in by_kind.values():
        mean_x = sum(x for x, _ in pairs) / len(pairs)
        mean_y = sum(y for _, y in pairs) / len(pairs)
        centered.extend((x - mean_x, y - mean_y) for x, y in pairs)
    # Each kind spends a degree of freedom on its own mean, so a slope needs
    # two points to spare beyond them.
    if len(points) - len(by_kind) < 2:
        return None
    spread = sum(x * x for x, _ in centered)
    if not spread:
        return None
    rise = sum(x * y for x, y in centered)
    return round(rise / spread * 1000, 1)


def tool_label(row):
    """Whether a tool round came before the reply's audio."""
    return "tool" if row["tool_batches"] else "no_tool"


def reply_kind(row):
    """What a reply answered and whether a tool round came before its audio."""
    return f"{row['origin']}_{tool_label(row)}"


def waterfall(first):
    """Consecutive segments from origin to the first native capture."""
    if not first or first.get("stage") != "native_audio_capture":
        return None
    accepted = first.get("elapsed_ms")
    output = first.get("first_provider_output_ms")
    end = first.get("provider_voice_end_ms")
    tool = first.get("first_tool_call_ms")
    responded = first.get("last_tool_response_ms")
    if accepted is None:
        return None
    if output is None or not 0 <= output <= accepted:
        output = accepted
    if end is not None and not 0 <= end <= output:
        end = None
    segments = {}

    # END can arrive with first output. The configured window supplies only
    # a baseline allowance, not evidence of when inference began.
    # A yield inside the window ends the candidate's turn in its place.
    window = first.get("silence_window_ms")
    yielded = first.get("candidate_yield_ms")
    cap = output if end is None else end
    if first.get("origin") == "last_voiced_pcm" and window:
        if yielded is not None and 0 <= yielded <= min(window, cap):
            segments["candidate_yield"] = yielded
        else:
            segments["silence_window"] = min(window, cap)
    base = segments.get("candidate_yield", segments.get("silence_window", 0))
    if end is not None:
        segments["endpointing"] = end - base
        base = end
    if (
        tool is not None
        and responded is not None
        and base <= tool <= responded <= accepted
    ):
        segments["tool_decision"] = tool - base
        segments["tool_rounds"] = responded - tool
        segments["post_tool_response"] = accepted - responded
    else:
        segments["response_start"] = output - base
        segments["output_to_audio"] = accepted - output
    segments["output_queue"] = first.get("output_queue_ms") or 0
    segments["native_capture"] = first.get("native_capture_ms") or 0
    total = first.get("server_total_ms")
    if total is not None:
        # Each field is rounded down to whole milliseconds separately.
        segments["unattributed"] = total - sum(segments.values())
    return segments


def analyze(lines, slow_ms=1500, stall_ms=5000):
    turns = {}
    rejected = Counter()
    for line in lines:
        try:
            record = parse(line)
        except Rejected as error:
            rejected[str(error)] += 1
            continue
        if record is None:
            continue
        key = (record["pid"], record["room"], record["session"], record["turn"])
        turns.setdefault(key, []).append(record)
    rows = []
    for key, events in turns.items():
        # Origins can move while a reply waits; event order must use the
        # session clock, not offsets from those different origins.
        def ordering(event):
            at = event.get("observed_ms")
            if at is None:
                at = offset(event)
            return (at is None, at or 0)

        events.sort(key=ordering)
        native = next(
            (
                e
                for e in events
                if e["stage"].startswith("native_audio_")
                or e["stage"] == "audio_enqueue_failed"
            ),
            None,
        )
        accepted = next((e for e in events if e["stage"] == "audio_accepted"), None)

        # The accepted record carries the handler's own split, logged after
        # the worker's copy was taken.
        first = {**(accepted or {}), **(native or {})} or None
        captured = bool(first and first.get("captured") is True)
        segments = waterfall(first) if captured else None
        total = first.get("server_total_ms") if captured else None
        response = None
        dominant = None
        if segments and total is not None:
            allowance = ("candidate_yield", "silence_window")
            response = total - sum(segments.get(name, 0) for name in allowance)
            spent = {
                name: ms
                for name, ms in segments.items()
                if name not in (*allowance, "unattributed")
            }
            dominant = max(spent, key=spent.get) if spent else None
        outcome = next(
            (
                e["stage"]
                for e in reversed(events)
                if e["stage"]
                in ("turn_complete", "interrupted", "paused", "held", "socket_replaced")
                and not (
                    e["stage"] == "socket_replaced"
                    and e.get("continuing", e.get("first_provider_output_ms") is None)
                )
            ),
            "open",
        )
        rows.append(
            {
                "pid": key[0],
                "room": key[1],
                "session": key[2],
                "turn": key[3],
                # Until output, a later input can replace the origin.
                "origin": (first or events[-1])["origin"],
                "outcome": outcome,
                # Rounds before the first audio, which is what held the reply
                # back; evidence recorded after it began is not.
                "tool_batches": (first.get("tool_batches") or 0)
                if first
                else max(e.get("tool_batches") or 0 for e in events),
                "first_audio": first,
                "captured": captured,
                "server_total_ms": total,
                "beyond_window_ms": response,
                "waterfall_ms": segments,
                "dominant": dominant,
                "slow": response is not None and response >= slow_ms,
                "stalled": response is not None and response >= stall_ms,
                "timeline": events,
            }
        )
    measured = [r for r in rows if r["captured"] and r["server_total_ms"] is not None]
    # Provider-event origins read near zero by construction.
    comparable = [r for r in measured if r["origin"] != "provider_event"]

    # A stall is a reply past `stall_ms`, a threshold set in the gap between
    # ordinary slow starts and sockets that stopped answering: it is counted as
    # a rate, and kept out of the distributions, where a few of them would set
    # every mean and tail.
    typical = [r for r in comparable if not r["stalled"]]

    # Whether replies slow as the conversation grows: the part of each wait
    # the model is responsible for, against the context it started from.
    by_context = [
        (reply_kind(r), r["first_audio"]["context_tokens"], r["beyond_window_ms"])
        for r in typical
        if r["first_audio"].get("context_tokens") is not None
    ]

    # On 3.1, END and the first output often arrive together, so the segment
    # named endpointing then also holds the model's own start.
    with_end = [
        r
        for r in comparable
        if r["first_audio"].get("provider_voice_end_ms") is not None
        and r["first_audio"].get("first_provider_output_ms") is not None
    ]
    return {
        "slow_ms": slow_ms,
        "stall_ms": stall_ms,
        "stalls": {
            "count": len(comparable) - len(typical),
            "of": len(comparable),
        },
        "end_with_output": {
            "count": sum(
                abs(
                    r["first_audio"]["first_provider_output_ms"]
                    - r["first_audio"]["provider_voice_end_ms"]
                )
                <= 10
                for r in with_end
            ),
            "of": len(with_end),
        },
        "turns": rows,
        "rejected_records": sum(rejected.values()),
        "rejected_reasons": dict(rejected),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "unmeasured": dict(
            Counter(
                (r["first_audio"] or {}).get("stage", "no_audio")
                for r in rows
                if not r["captured"]
            )
        ),
        # Replies the statistics never see, by whether a tool round came
        # first: when most are tool turns, the distributions read fast.
        "unmeasured_by_tool": dict(
            Counter(tool_label(r) for r in rows if not r["captured"])
        ),
        "server_total_by_origin": {
            origin: statistics(
                [
                    r["server_total_ms"]
                    for r in measured
                    if r["origin"] == origin and not r["stalled"]
                ]
            )
            for origin in sorted({r["origin"] for r in measured})
        },
        "beyond_window": statistics([r["beyond_window_ms"] for r in typical]),
        # The same with the stalls left in, so the tail they set stays visible.
        "beyond_window_with_stalls": statistics(
            [r["beyond_window_ms"] for r in comparable]
        ),
        # With an acknowledgement played during a tool round, the candidate
        # hears something before the reply: how much sooner. The clip went out
        # on an empty queue, so its queueing is its capture; the reply's is its
        # native capture, which waited behind the clip when the clip was longer.
        "tool_ack_lead": statistics(
            [
                r["server_total_ms"] - r["first_audio"]["tool_ack_ms"]
                for r in typical
                if r["first_audio"].get("tool_ack_ms") is not None
            ]
        ),
        "beyond_window_by_kind": {
            kind: statistics(
                [r["beyond_window_ms"] for r in typical if reply_kind(r) == kind]
            )
            for kind in sorted({reply_kind(r) for r in typical})
        },
        # How long before END the first input words arrived: the room an
        # early end-of-turn decision would have to judge a finished sentence.
        "transcript_before_end": statistics(
            [
                r["first_audio"]["provider_voice_end_ms"]
                - r["first_audio"]["first_transcript_ms"]
                for r in typical
                if r["first_audio"].get("provider_voice_end_ms") is not None
                and r["first_audio"].get("first_transcript_ms") is not None
            ]
        ),
        "beyond_window_by_context": {
            band: statistics(
                [ms for _, tokens, ms in by_context if context_band(tokens) == band]
            )
            for band in sorted(
                {context_band(tokens) for _, tokens, _ in by_context},
                key=lambda band: int(band.split("k_")[0]),
            )
        },
        "context_slope_ms_per_1k": slope_per_1k(by_context),
        "tool_calls": dict(
            Counter(
                name
                for r in rows
                for e in r["timeline"]
                if e["stage"] == "tool_response" and e.get("sent") is not False
                for name in e.get("tool_names") or ()
            )
        ),
        "segments": {
            name: statistics(
                [r["waterfall_ms"][name] for r in typical if name in r["waterfall_ms"]]
            )
            for name in SEGMENTS
        },
        "dominant": dict(Counter(r["dominant"] for r in typical if r["dominant"])),
        "slow_turns": [turn_id(r) for r in comparable if r["slow"]],
        "worst": sorted(comparable, key=lambda r: -r["beyond_window_ms"])[:5],
    }


def turn_id(row):
    return f"{row['room']}/s{row['session']}/t{row['turn']}"


def verdict(report):
    if not report["stalls"]["of"]:
        return "NO_DATA"
    return "SLOW" if report["slow_turns"] else "HEALTHY"


def bar(ms, total, width=30):
    if not total or ms <= 0:
        return ""
    return "#" * max(1, round(width * ms / total))


def render(report):
    lines = [
        "Server reply latency, ms from each reply's origin (monotonic clock).",
        "Native capture is not browser playback; provider internals are unmeasured.",
    ]
    for row in report["turns"]:
        lines.append(
            f"\n{turn_id(row)} socket={row['timeline'][-1]['socket']} "
            f"origin={row['origin']} outcome={row['outcome']}"
            + (f" tool_batches={row['tool_batches']}" if row["tool_batches"] else "")
            + ("  SLOW" if row["slow"] else "")
        )
        segments = row["waterfall_ms"]
        if segments:
            total = row["server_total_ms"]
            lines.append(
                f"  total {total}  beyond window {row['beyond_window_ms']}  "
                f"dominant {row['dominant']}"
            )
            for name in (*SEGMENTS, "unattributed"):
                if name in segments:
                    ms = segments[name]
                    lines.append(f"  {name:<19}{ms:>7}  {bar(ms, total)}")
            first = row["first_audio"]
            overlays = [
                f"{name}={first[name]}"
                for name in (
                    "tool_handler_ms",
                    "provider_event_queue_ms",
                    "stale_drop_ms",
                    "enqueue_ms",
                    "handler_tail_ms",
                    "output_queue_frames",
                    "playout_ahead_ms",
                    "context_tokens",
                )
                if first.get(name) is not None
                and (name != "tool_handler_ms" or row["tool_batches"])
            ]
            if overlays:
                lines.append("  within: " + " ".join(overlays))
        else:
            stage = (row["first_audio"] or {}).get("stage", "no audio")
            lines.append(f"  first audio not captured ({stage})")
        for event in row["timeline"]:
            at = offset(event)
            detail = ""
            if event["stage"] == "tool_response":
                detail = " " + " ".join(
                    f"{name}={event.get(name)}"
                    for name in (
                        "tool_count",
                        "tool_names",
                        "tool_batch_ms",
                        "tool_execute_ms",
                        "board_resend_ms",
                        "tool_send_ms",
                        "sent",
                    )
                )
            lines.append(f"    +{'?' if at is None else at} {event['stage']}{detail}")
    lines.append("")
    lines.extend(summary_lines(report))
    return "\n".join(lines)


def stat_line(label, stats):
    line = f"{label}=n{stats['count']} p50={stats['p50_ms']} p95={stats['p95_ms']}"
    return line + (" small_n" if stats["small_sample"] else "")


def summary_lines(report):
    out = [
        f"verdict={verdict(report)}",
        f"turns={len(report['turns'])}",
        f"captured={sum(r['captured'] for r in report['turns'])}",
        f"slow_ms={report['slow_ms']}",
        f"slow_turns={len(report['slow_turns'])}",
        f"stalls={report['stalls']['count']}/{report['stalls']['of']}",
        f"end_with_output={report['end_with_output']['count']}"
        f"/{report['end_with_output']['of']}",
        f"rejected_records={report['rejected_records']}",
    ]
    for name, count in sorted(report["outcomes"].items()):
        out.append(f"outcome_{name}={count}")
    for name, count in sorted(report["unmeasured"].items()):
        out.append(f"unmeasured_{name}={count}")
    for name, count in sorted(report["unmeasured_by_tool"].items()):
        out.append(f"unmeasured_by_tool_{name}={count}")
    for origin, stats in report["server_total_by_origin"].items():
        if stats["count"]:
            out.append(stat_line(f"total_{origin}", stats))
    for label in (
        "beyond_window",
        "beyond_window_with_stalls",
        "transcript_before_end",
        "tool_ack_lead",
    ):
        if report[label]["count"]:
            out.append(stat_line(label, report[label]))
    for kind, stats in report["beyond_window_by_kind"].items():
        out.append(stat_line(f"kind_{kind}", stats))
    for band, stats in report["beyond_window_by_context"].items():
        out.append(stat_line(f"context_{band}", stats))
    if report["context_slope_ms_per_1k"] is not None:
        out.append(f"context_slope_ms_per_1k={report['context_slope_ms_per_1k']}")
    for name, count in sorted(report["tool_calls"].items()):
        out.append(f"tool_{name}={count}")
    for name in SEGMENTS:
        if report["segments"][name]["count"]:
            out.append(stat_line(f"segment_{name}", report["segments"][name]))
    for name, count in sorted(report["dominant"].items(), key=lambda i: -i[1]):
        out.append(f"dominant_{name}={count}")
    if report["worst"]:
        worst = report["worst"][0]
        out.append(
            f"worst={turn_id(worst)} beyond_window={worst['beyond_window_ms']} "
            f"dominant={worst['dominant']}"
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", help="server log; omit to read stdin")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="full report as JSON")
    output.add_argument(
        "--summary", action="store_true", help="key=value lines only, for scripts"
    )
    parser.add_argument(
        "--slow-ms",
        type=int,
        default=1500,
        help="flag a reply whose time beyond the silence window reaches this",
    )
    parser.add_argument(
        "--stall-ms",
        type=int,
        default=5000,
        help="count a reply this far beyond the window as a stall, not a slow reply",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail on rejected records or when no input-to-capture reply was measured",
    )
    args = parser.parse_args()
    if args.log:
        with open(args.log, encoding="utf-8", errors="replace") as stream:
            report = analyze(stream, args.slow_ms, args.stall_ms)
    else:
        report = analyze(sys.stdin, args.slow_ms, args.stall_ms)
    if args.json:
        print(json.dumps(report, indent=2))
    elif args.summary:
        print("\n".join(summary_lines(report)))
    else:
        print(render(report))
    if not report["turns"]:
        return 1
    if args.strict and (report["rejected_records"] or verdict(report) == "NO_DATA"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
