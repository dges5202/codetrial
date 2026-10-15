import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze-latency.py"
SPEC = importlib.util.spec_from_file_location("analyze_latency", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def line(**fields):
    return "Oct 14 agent[12]: codetrial latency " + json.dumps(
        {
            "event": "reply_latency",
            "version": 2,
            "pid": 7,
            "room": "r",
            "session": 1,
            "socket": 1,
            "origin_socket": 1,
            "turn": 1,
            "origin": "last_voiced_pcm",
            "silence_window_ms": 3_000,
            **fields,
        }
    )


def captured(**fields):
    return line(
        stage="native_audio_capture",
        captured=True,
        **{
            "elapsed_ms": 3_700,
            "provider_voice_end_ms": 3_100,
            "first_provider_output_ms": 3_500,
            "output_queue_ms": 20,
            "native_capture_ms": 10,
            "server_total_ms": 3_730,
            **fields,
        },
    )


def run(*args, stdin):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


class WaterfallTests(unittest.TestCase):
    def test_plain_reply_partitions_the_server_total(self):
        row = MODULE.analyze([captured()])["turns"][0]
        self.assertEqual(
            row["waterfall_ms"],
            {
                "silence_window": 3_000,
                "endpointing": 100,
                "response_start": 400,
                "output_to_audio": 200,
                "output_queue": 20,
                "native_capture": 10,
                "unattributed": 0,
            },
        )
        self.assertEqual(row["beyond_window_ms"], 730)
        self.assertEqual(row["dominant"], "response_start")

    def test_an_end_sent_with_the_first_output_still_shows_the_wait(self):
        row = MODULE.analyze(
            [captured(provider_voice_end_ms=3_500, first_provider_output_ms=3_500)]
        )["turns"][0]
        self.assertEqual(row["waterfall_ms"]["endpointing"], 500)
        self.assertEqual(row["waterfall_ms"]["response_start"], 0)
        self.assertEqual(row["dominant"], "endpointing")

    def test_the_window_applies_only_after_voiced_input(self):
        row = MODULE.analyze([captured(origin="prompt", provider_voice_end_ms=None)])[
            "turns"
        ][0]
        self.assertNotIn("silence_window", row["waterfall_ms"])
        self.assertEqual(row["beyond_window_ms"], 3_730)

    def test_handler_split_comes_from_the_accepted_record(self):
        row = MODULE.analyze(
            [
                line(stage="audio_accepted", elapsed_ms=3_700, handler_tail_ms=40),
                captured(),
            ]
        )["turns"][0]
        self.assertEqual(row["first_audio"]["handler_tail_ms"], 40)
        self.assertEqual(row["first_audio"]["stage"], "native_audio_capture")

    def test_a_yield_inside_the_window_replaces_it(self):
        row = MODULE.analyze(
            [
                captured(
                    candidate_yield_ms=800,
                    provider_voice_end_ms=1_200,
                    first_provider_output_ms=1_200,
                    elapsed_ms=1_400,
                    server_total_ms=1_430,
                )
            ]
        )["turns"][0]
        self.assertEqual(row["waterfall_ms"]["candidate_yield"], 800)
        self.assertEqual(row["waterfall_ms"]["endpointing"], 400)
        self.assertNotIn("silence_window", row["waterfall_ms"])
        self.assertEqual(row["beyond_window_ms"], 630)
        self.assertEqual(row["dominant"], "endpointing")

    def test_a_yield_after_the_end_of_speech_changes_nothing(self):
        row = MODULE.analyze([captured(candidate_yield_ms=3_300)])["turns"][0]
        self.assertNotIn("candidate_yield", row["waterfall_ms"])
        self.assertEqual(row["waterfall_ms"]["silence_window"], 3_000)

    def test_tool_turn_splits_decision_rounds_and_continuation(self):
        row = MODULE.analyze(
            [
                line(stage="tool_response", elapsed_ms=3_900, tool_batch_ms=40),
                captured(
                    elapsed_ms=5_000,
                    first_tool_call_ms=3_400,
                    last_tool_response_ms=3_900,
                    first_provider_output_ms=3_400,
                    tool_batches=1,
                    tool_handler_ms=40,
                    server_total_ms=5_030,
                ),
            ]
        )["turns"][0]
        self.assertEqual(row["waterfall_ms"]["tool_decision"], 300)
        self.assertEqual(row["waterfall_ms"]["tool_rounds"], 500)
        self.assertEqual(row["waterfall_ms"]["post_tool_response"], 1_100)
        self.assertNotIn("response_start", row["waterfall_ms"])
        self.assertEqual(row["dominant"], "post_tool_response")
        self.assertEqual(
            [e["stage"] for e in row["timeline"]],
            ["tool_response", "native_audio_capture"],
        )

    def test_tool_names_and_transcript_lead_are_summarized(self):
        report = MODULE.analyze(
            [
                line(
                    stage="tool_response",
                    elapsed_ms=3_900,
                    tool_names=["record_framework_evidence", "log_hint"],
                    sent=True,
                ),
                line(
                    stage="tool_response",
                    elapsed_ms=3_950,
                    tool_names=["log_hint"],
                    sent=False,
                ),
                captured(first_transcript_ms=1_100),
            ]
        )
        self.assertEqual(
            report["tool_calls"], {"record_framework_evidence": 1, "log_hint": 1}
        )
        self.assertEqual(report["transcript_before_end"]["p50_ms"], 2_000)
        self.assertEqual(
            MODULE.analyze([line(stage="turn_complete", tool_names="x")])[
                "rejected_records"
            ],
            1,
        )

    def test_context_size_is_optional_but_must_be_a_count(self):
        self.assertEqual(
            MODULE.analyze([captured(context_tokens=None)])["rejected_records"], 0
        )
        self.assertEqual(
            MODULE.analyze([captured(context_tokens=5_000)])["rejected_records"], 0
        )
        self.assertEqual(
            MODULE.analyze([captured(context_tokens="big")])["rejected_records"], 1
        )

    def test_audio_before_a_tool_call_is_not_a_tool_turn(self):
        row = MODULE.analyze(
            [captured(first_tool_call_ms=4_000, last_tool_response_ms=4_100)]
        )["turns"][0]
        self.assertIn("response_start", row["waterfall_ms"])

    def test_an_end_after_output_is_not_endpointing(self):
        row = MODULE.analyze([captured(provider_voice_end_ms=3_600)])["turns"][0]
        self.assertNotIn("endpointing", row["waterfall_ms"])
        self.assertEqual(row["waterfall_ms"]["response_start"], 500)


class AggregateTests(unittest.TestCase):
    def test_reply_time_is_banded_and_sloped_by_context_size(self):
        report = MODULE.analyze(
            [
                captured(turn=1, context_tokens=4_000),
                captured(turn=2, context_tokens=12_000, server_total_ms=3_830),
                captured(turn=3, context_tokens=20_000, server_total_ms=3_930),
                captured(turn=4, context_tokens=None),
            ]
        )
        self.assertEqual(
            {
                band: stats["count"]
                for band, stats in report["beyond_window_by_context"].items()
            },
            {"0k_8k": 1, "8k_16k": 1, "16k_32k": 1},
        )
        self.assertEqual(
            list(report["beyond_window_by_context"]), ["0k_8k", "8k_16k", "16k_32k"]
        )
        self.assertEqual(report["context_slope_ms_per_1k"], 12.5)
        summary = MODULE.summary_lines(report)
        self.assertIn("context_8k_16k=n1 p50=830 p95=830 small_n", summary)
        self.assertIn("context_slope_ms_per_1k=12.5", summary)

    def test_a_flat_or_thin_context_sample_has_no_slope(self):
        self.assertIsNone(MODULE.slope_per_1k([("a", 1_000, 5), ("a", 2_000, 6)]))
        self.assertIsNone(
            MODULE.slope_per_1k([("a", 1_000, 5), ("a", 1_000, 6), ("a", 1_000, 7)])
        )
        # Three points, but the lone "b" only fixes its own mean: two left.
        self.assertIsNone(
            MODULE.slope_per_1k([("a", 1_000, 5), ("a", 2_000, 6), ("b", 9_000, 40)])
        )
        self.assertEqual(MODULE.context_band(64_000), "64k_up")

    def test_the_context_slope_is_fitted_within_each_kind_of_reply(self):
        # Tool replies are slower and come later, but neither kind slows as
        # its own context grows: a pooled fit would call that mix growth.
        points = [
            ("no_tool", 4_000, 600),
            ("no_tool", 8_000, 600),
            ("tool", 20_000, 1_300),
            ("tool", 24_000, 1_300),
        ]
        self.assertEqual(MODULE.slope_per_1k(points), 0.0)

    def test_percentiles_interpolate_and_a_small_sample_says_so(self):
        stats = MODULE.statistics(list(range(1, 21)))
        # Half a rank is kept, not rounded to the even neighbour.
        self.assertEqual(stats["p50_ms"], 10.5)
        self.assertEqual(stats["p95_ms"], 19.1)
        self.assertFalse(stats["small_sample"])
        thin = MODULE.statistics([100, 200, 900])
        self.assertEqual((thin["p50_ms"], thin["p95_ms"]), (200, 830))
        self.assertTrue(thin["small_sample"])

    def test_stalls_are_a_rate_not_part_of_the_distribution(self):
        report = MODULE.analyze(
            [
                captured(turn=1),
                captured(turn=2, tool_batches=1, server_total_ms=4_730),
                captured(turn=3, server_total_ms=12_000, elapsed_ms=11_970),
            ]
        )
        self.assertEqual(report["stalls"], {"count": 1, "of": 3})
        self.assertEqual(report["beyond_window"]["count"], 2)
        self.assertEqual(sum(report["dominant"].values()), 2)
        self.assertEqual(
            report["server_total_by_origin"]["last_voiced_pcm"]["max_ms"], 4_730
        )
        self.assertEqual(report["beyond_window"]["max_ms"], 1_730)
        self.assertEqual(
            {
                kind: stats["count"]
                for kind, stats in report["beyond_window_by_kind"].items()
            },
            {"last_voiced_pcm_no_tool": 1, "last_voiced_pcm_tool": 1},
        )
        summary = MODULE.summary_lines(report)
        self.assertIn("stalls=1/3", summary)
        self.assertIn("kind_last_voiced_pcm_tool=n1 p50=1730 p95=1730 small_n", summary)
        # The stall is still the worst reply, so it stays findable.
        self.assertEqual(MODULE.turn_id(report["worst"][0]), "r/s1/t3")

    def test_unmeasured_replies_are_split_by_tool_round(self):
        report = MODULE.analyze(
            [
                line(
                    stage="native_audio_cancelled",
                    turn=1,
                    captured=False,
                    tool_batches=1,
                ),
                line(stage="turn_complete", turn=2, elapsed_ms=2_000),
            ]
        )
        self.assertEqual(report["unmeasured_by_tool"], {"tool": 1, "no_tool": 1})

    def test_an_acknowledgement_reports_how_much_sooner_something_was_heard(self):
        report = MODULE.analyze(
            [
                captured(
                    turn=1, tool_batches=1, first_tool_call_ms=3_150, tool_ack_ms=3_160
                ),
                captured(turn=2),
            ]
        )
        self.assertEqual(report["tool_ack_lead"]["count"], 1)
        # To the reply's capture, not its acceptance: queued behind the clip,
        # the reply is heard only when the clip has played.
        self.assertEqual(report["tool_ack_lead"]["p50_ms"], 570)
        self.assertIn(
            "tool_ack_lead=n1 p50=570 p95=570 small_n", MODULE.summary_lines(report)
        )

    def test_tool_rounds_after_the_first_audio_do_not_make_a_tool_reply(self):
        report = MODULE.analyze(
            [
                captured(turn=1, tool_batches=0),
                line(stage="turn_complete", turn=1, tool_batches=1, elapsed_ms=5_000),
            ]
        )
        self.assertEqual(report["turns"][0]["tool_batches"], 0)
        self.assertEqual(
            list(report["beyond_window_by_kind"]), ["last_voiced_pcm_no_tool"]
        )

    def test_a_sample_of_only_stalls_is_slow_not_missing(self):
        report = MODULE.analyze([captured(server_total_ms=12_000, elapsed_ms=11_970)])
        self.assertEqual(report["beyond_window"]["count"], 0)
        self.assertEqual(MODULE.verdict(report), "SLOW")
        # An origin with nothing but stalls has no total to print, not a crash.
        summary = MODULE.summary_lines(report)
        self.assertFalse(any(line.startswith("total_") for line in summary))
        self.assertIn("beyond_window_with_stalls=n1 p50=9000 p95=9000 small_n", summary)

    def test_an_end_arriving_with_the_first_output_is_counted(self):
        report = MODULE.analyze(
            [
                captured(
                    turn=1, provider_voice_end_ms=3_500, first_provider_output_ms=3_505
                ),
                captured(turn=2),
            ]
        )
        self.assertEqual(report["end_with_output"], {"count": 1, "of": 2})
        self.assertIn("end_with_output=1/2", MODULE.summary_lines(report))

    def test_records_join_across_sockets_and_keep_unmeasured_turns(self):
        report = MODULE.analyze(
            [
                line(stage="socket_replaced", socket=2, elapsed_ms=900),
                captured(socket=2),
                line(stage="turn_complete", elapsed_ms=4_000, socket=2),
                line(stage="native_audio_cancelled", turn=2, captured=False),
                line(stage="interrupted", turn=2, elapsed_ms=50),
                line(stage="turn_complete", turn=3, elapsed_ms=2_000),
            ]
        )
        first, cancelled, silent = report["turns"]
        self.assertEqual(
            [e["stage"] for e in first["timeline"]],
            ["socket_replaced", "native_audio_capture", "turn_complete"],
        )
        self.assertEqual(first["outcome"], "turn_complete")
        self.assertEqual(cancelled["outcome"], "interrupted")
        self.assertFalse(cancelled["captured"])
        self.assertEqual(
            report["unmeasured"], {"native_audio_cancelled": 1, "no_audio": 1}
        )
        self.assertEqual(
            report["server_total_by_origin"]["last_voiced_pcm"]["count"], 1
        )

    def test_replacement_of_an_unstarted_reply_keeps_the_outcome_open(self):
        report = MODULE.analyze([line(stage="socket_replaced", elapsed_ms=5_000)])
        self.assertEqual(report["turns"][0]["outcome"], "open")
        tool_wait = MODULE.analyze(
            [
                line(
                    stage="socket_replaced",
                    elapsed_ms=5_000,
                    first_provider_output_ms=1_000,
                    continuing=True,
                )
            ]
        )
        self.assertEqual(tool_wait["turns"][0]["outcome"], "open")

    def test_receipt_before_the_first_local_input_has_a_signed_session_time(self):
        report = MODULE.analyze(
            [
                line(
                    origin="provider_event",
                    stage="turn_complete",
                    elapsed_ms=5,
                    observed_ms=-5,
                ),
            ]
        )
        self.assertEqual(report["turns"][0]["outcome"], "turn_complete")
        self.assertEqual(report["turns"][0]["timeline"][0]["observed_ms"], -5)

    def test_moving_the_origin_does_not_move_a_reconnect_after_completion(self):
        report = MODULE.analyze(
            [
                line(stage="socket_replaced", elapsed_ms=5_000, observed_ms=5_000),
                captured(elapsed_ms=1_000, server_total_ms=1_030, observed_ms=7_030),
                line(stage="turn_complete", elapsed_ms=2_000, observed_ms=8_000),
            ]
        )
        row = report["turns"][0]
        self.assertEqual(row["outcome"], "turn_complete")
        self.assertEqual(
            [e["stage"] for e in row["timeline"]],
            ["socket_replaced", "native_audio_capture", "turn_complete"],
        )

    def test_an_enqueue_failure_is_counted_without_becoming_a_latency_sample(self):
        report = MODULE.analyze(
            [
                line(
                    stage="audio_enqueue_failed",
                    captured=False,
                    elapsed_ms=500,
                    server_total_ms=510,
                    enqueued=False,
                )
            ]
        )
        self.assertEqual(report["unmeasured"], {"audio_enqueue_failed": 1})
        self.assertEqual(report["beyond_window"]["count"], 0)
        self.assertEqual(MODULE.verdict(report), "NO_DATA")

    def test_percentiles_exclude_provider_event_origins(self):
        report = MODULE.analyze(
            [
                captured(turn=n, server_total_ms=3_030 + n, elapsed_ms=3_000 + n)
                for n in range(1, 21)
            ]
            + [captured(turn=21, origin="provider_event", server_total_ms=5)]
        )
        self.assertEqual(report["beyond_window"]["count"], 20)
        self.assertEqual(report["beyond_window"]["p95_ms"], 49)
        self.assertEqual(report["server_total_by_origin"]["provider_event"]["count"], 1)

    def test_slow_turns_are_judged_after_the_end_of_speech(self):
        report = MODULE.analyze(
            [captured(), captured(turn=2, elapsed_ms=5_200, server_total_ms=5_230)],
            slow_ms=1_500,
        )
        self.assertEqual(report["slow_turns"], ["r/s1/t2"])
        self.assertEqual(report["worst"][0]["turn"], 2)
        self.assertEqual(MODULE.verdict(report), "SLOW")

    def test_incomplete_and_unbounded_capture_records_do_not_crash_aggregation(self):
        report = MODULE.analyze(
            [
                captured(elapsed_ms=None),
                captured(server_total_ms=None),
                captured(tool_batches=[]),
                captured(silence_window_ms=-1),
                line(stage="native_audio_capture", captured="yes"),
                captured(server_total_ms=10**500),
            ]
        )
        self.assertEqual(report["rejected_records"], 6)
        self.assertEqual(report["turns"], [])

    def test_false_capture_stages_and_deeply_nested_json_are_rejected(self):
        report = MODULE.analyze(
            [
                line(
                    stage="native_audio_cancelled", captured=True, server_total_ms=500
                ),
                line(stage="audio_accepted", captured=True, server_total_ms=500),
                "codetrial latency " + "[" * 2_000 + "]" * 2_000,
            ]
        )
        self.assertEqual(report["rejected_records"], 3)
        self.assertEqual(report["turns"], [])

    def test_provider_only_captures_do_not_claim_healthy_input_latency(self):
        report = MODULE.analyze([captured(origin="provider_event")])
        self.assertTrue(report["turns"][0]["captured"])
        self.assertEqual(report["beyond_window"]["count"], 0)
        self.assertEqual(MODULE.verdict(report), "NO_DATA")

    def test_invalid_records_are_rejected_not_crashed_on(self):
        report = MODULE.analyze(
            [
                "codetrial latency {",
                line(version=1, stage="turn_complete"),
                line(stage="turn_complete", origin=["bad"]),
                line(stage="made_up"),
                line(stage="turn_complete", elapsed_ms=True),
                captured(server_total_ms=float("nan")),
                captured(server_total_ms=-4),
                line(stage="turn_complete", session="1"),
            ]
        )
        self.assertEqual(report["turns"], [])
        self.assertEqual(report["rejected_records"], 8)


class CliTests(unittest.TestCase):
    def test_no_records_fails(self):
        result = run("--summary", stdin="unrelated\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("verdict=NO_DATA", result.stdout)

    def test_a_truncated_line_fails_only_when_strict(self):
        stdin = "\n".join([captured(), 'codetrial latency {"event"']) + "\n"
        self.assertEqual(run("--summary", stdin=stdin).returncode, 0)
        self.assertEqual(run("--summary", "--strict", stdin=stdin).returncode, 1)

    def test_text_and_json_reports(self):
        text = run(stdin=captured() + "\n")
        self.assertEqual(text.returncode, 0, text.stderr)
        self.assertIn("silence_window", text.stdout)
        self.assertIn("verdict=HEALTHY", text.stdout)
        report = json.loads(run("--json", stdin=captured() + "\n").stdout)
        self.assertEqual(sum(report["turns"][0]["waterfall_ms"].values()), 3_730)


if __name__ == "__main__":
    unittest.main()
