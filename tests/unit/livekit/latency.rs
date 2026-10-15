use super::*;
use std::time::Duration;

fn ms(start: Instant, offset: u64) -> Instant {
    start + Duration::from_millis(offset)
}

fn traced() -> (Instant, ReplyLatency) {
    let mut trace = ReplyLatency::default();
    trace.bind("room", 4, 1, 3_000);
    (Instant::now(), trace)
}

#[test]
fn milestones_are_offsets_from_the_last_voiced_frame() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.transcript(ms(start, 10));
    trace.voice(ms(start, 20));
    trace.voice_end(ms(start, 200));
    trace.output(ms(start, 500), false);
    let record = trace.audio(ms(start, 800)).unwrap();
    assert_eq!(record["origin"], "last_voiced_pcm");
    assert_eq!(record["elapsed_ms"], 780);
    assert_eq!(record["provider_voice_end_ms"], 180);
    assert_eq!(record["first_provider_output_ms"], 480);
    // Seen before the origin moved: early, not missing.
    assert_eq!(record["latest_transcript_ms"], -10);
    assert_eq!(record["first_tool_call_ms"], Value::Null);
    assert_eq!(record["silence_window_ms"], 3_000);
    assert_eq!(
        (record["session"].clone(), record["socket"].clone()),
        (4.into(), 1.into())
    );
    assert!(trace.audio(ms(start, 900)).is_none());
}

/// A new utterance with no transcript of its own reports none, not the
/// previous utterance's words against its own origin.
#[test]
fn a_new_utterance_drops_the_last_ones_transcript() {
    let (start, mut trace) = traced();
    trace.voice_start();
    trace.voice(start);
    trace.transcript(ms(start, 10));
    trace.voice_end(ms(start, 100));
    trace.voice_start();
    trace.voice(ms(start, 1_000));
    trace.voice_end(ms(start, 1_100));
    trace.output(ms(start, 1_500), false);
    let record = trace.audio(ms(start, 1_600)).unwrap();
    assert_eq!(record["elapsed_ms"], 600);
    assert_eq!(record["latest_transcript_ms"], Value::Null);
    assert_eq!(record["first_transcript_ms"], Value::Null);
}

#[test]
fn every_turn_keeps_its_acoustic_origin_after_the_first() {
    let (start, mut trace) = traced();
    for turn in 0..3 {
        let base = 10_000 * turn;
        trace.voice(ms(start, base));
        let record = trace.audio(ms(start, base + 700)).unwrap();
        assert_eq!(record["turn"], turn + 1);
        assert_eq!(record["origin"], "last_voiced_pcm");
        assert_eq!(record["elapsed_ms"], 700);
        let done = trace.boundary(Boundary::TurnComplete, ms(start, base + 2_000));
        assert_eq!(done.unwrap()["stage"], "turn_complete");
    }
}

#[test]
fn the_latest_input_before_output_is_what_the_reply_answers() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.prompt(ms(start, 30_000));
    let record = trace.audio(ms(start, 30_400)).unwrap();
    assert_eq!(record["origin"], "prompt");
    assert_eq!(record["elapsed_ms"], 400);
    trace.boundary(Boundary::TurnComplete, ms(start, 31_000));

    trace.prompt(ms(start, 40_000));
    trace.voice(ms(start, 45_000));
    let record = trace.audio(ms(start, 45_600)).unwrap();
    assert_eq!(record["origin"], "last_voiced_pcm");
    assert_eq!(record["elapsed_ms"], 600);
}

#[test]
fn a_prompt_sent_behind_a_generation_times_the_next_reply() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.audio(ms(start, 500));
    trace.prompt(ms(start, 900));
    trace.boundary(Boundary::TurnComplete, ms(start, 1_000));
    let record = trace.audio(ms(start, 1_400)).unwrap();
    assert_eq!(record["turn"], 2);
    assert_eq!(record["origin"], "prompt");
    assert_eq!(record["elapsed_ms"], 500);
}

#[test]
fn speech_over_a_finishing_reply_survives_either_ending() {
    for ending in [Boundary::TurnComplete, Boundary::Interrupted] {
        let (start, mut trace) = traced();
        trace.voice(start);
        trace.audio(ms(start, 50));
        trace.voice(ms(start, 100));
        trace.voice(ms(start, 120));
        trace.voice_end(ms(start, 125));
        assert!(trace.audio(ms(start, 130)).is_none());
        assert!(trace.boundary(ending, ms(start, 140)).is_some());
        let record = trace.audio(ms(start, 320)).unwrap();
        assert_eq!(record["turn"], 2);
        assert_eq!(record["origin"], "last_voiced_pcm");
        assert_eq!(record["elapsed_ms"], 200);
        assert_eq!(record["provider_voice_end_ms"], 5);
    }
}

#[test]
fn frames_after_the_end_of_speech_do_not_move_the_origin_until_a_new_start() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.voice_end(ms(start, 100));
    trace.voice(ms(start, 150));
    trace.output(ms(start, 600), false);
    let record = trace.audio(ms(start, 700)).unwrap();
    assert_eq!(record["elapsed_ms"], 700);
    assert_eq!(record["provider_voice_end_ms"], 100);
    trace.boundary(Boundary::TurnComplete, ms(start, 800));

    trace.voice(ms(start, 1_000));
    trace.voice_end(ms(start, 1_100));
    trace.voice_start();
    trace.voice(ms(start, 1_500));
    let record = trace.audio(ms(start, 1_900)).unwrap();
    assert_eq!(record["elapsed_ms"], 400);
    assert_eq!(record["provider_voice_end_ms"], Value::Null);
}

#[test]
fn an_unstarted_reply_waits_through_interruptions_and_socket_replacement() {
    let (start, mut trace) = traced();
    trace.voice(start);
    assert!(
        trace
            .boundary(Boundary::Interrupted, ms(start, 100))
            .is_none()
    );
    trace.set_socket(2);
    let stalled = trace
        .boundary(Boundary::SocketReplaced, ms(start, 5_000))
        .unwrap();
    assert_eq!(stalled["stage"], "socket_replaced");
    assert_eq!(stalled["elapsed_ms"], 5_000);
    let record = trace.audio(ms(start, 6_000)).unwrap();
    assert_eq!(record["turn"], 1);
    assert_eq!(record["elapsed_ms"], 6_000);
    assert_eq!(
        (record["origin_socket"].clone(), record["socket"].clone()),
        (1.into(), 2.into())
    );
    let closed = trace.boundary(Boundary::SocketReplaced, ms(start, 7_000));
    assert_eq!(closed.unwrap()["stage"], "socket_replaced");
    assert!(trace.audio(ms(start, 8_000)).unwrap()["origin"] == "provider_event");
}

#[test]
fn a_local_cut_closes_the_turn_even_before_output() {
    let (start, mut trace) = traced();
    trace.voice(start);
    let paused = trace.boundary(Boundary::Paused, ms(start, 100)).unwrap();
    assert_eq!(paused["stage"], "paused");
    assert!(
        trace
            .boundary(Boundary::TurnComplete, ms(start, 200))
            .is_none()
    );
    trace.prompt(ms(start, 10_000));
    let record = trace.audio(ms(start, 10_300)).unwrap();
    assert_eq!(record["turn"], 2);
    assert_eq!(record["elapsed_ms"], 300);
}

#[test]
fn tool_turns_split_decision_handling_and_continuation() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.voice_end(ms(start, 1_000));
    trace.output(ms(start, 1_400), true);
    let batch = trace.tool_response(ms(start, 1_450), 50, true);
    assert_eq!(batch["stage"], "tool_response");
    assert_eq!(batch["last_tool_response_ms"], 1_450);
    trace.output(ms(start, 2_000), true);
    trace.tool_response(ms(start, 2_100), 100, true);
    let record = trace.audio(ms(start, 2_900)).unwrap();
    assert_eq!(record["first_provider_output_ms"], 1_400);
    assert_eq!(record["first_tool_call_ms"], 1_400);
    assert_eq!(record["last_tool_response_ms"], 2_100);
    assert_eq!(record["tool_batches"], 2);
    assert_eq!(record["tool_handler_ms"], 150);
}

#[test]
fn input_with_no_voiced_frame_falls_back_without_claiming_acoustic_latency() {
    let (start, mut trace) = traced();
    trace.transcript(start);
    let record = trace.audio(ms(start, 50)).unwrap();
    assert_eq!(record["origin"], "transcript");
    trace.boundary(Boundary::TurnComplete, ms(start, 60));
    trace.voice_end(ms(start, 100));
    assert_eq!(
        trace.audio(ms(start, 150)).unwrap()["origin"],
        "provider_voice_end"
    );
}

#[test]
fn recovery_briefings_preserve_the_original_wait_through_multiple_replacements() {
    let (start, mut trace) = traced();
    trace.voice(start);
    for socket in 2..=3 {
        trace.set_socket(socket);
        trace.boundary(Boundary::SocketReplaced, ms(start, socket * 5_000));
        trace.recovery_prompt(ms(start, socket * 5_000 + 200));
    }
    let record = trace.audio(ms(start, 20_000)).unwrap();
    assert_eq!(record["origin"], "last_voiced_pcm");
    assert_eq!(record["elapsed_ms"], 20_000);
    assert_eq!(record["origin_socket"], 1);
    assert_eq!(record["socket"], 3);
}

#[test]
fn transcription_of_the_current_reply_does_not_create_a_pending_reply() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.audio(ms(start, 300));
    trace.transcript(ms(start, 400));
    trace.boundary(Boundary::TurnComplete, ms(start, 500));
    let record = trace.audio(ms(start, 30_000)).unwrap();
    assert_eq!(record["origin"], "provider_event");
    assert_eq!(record["elapsed_ms"], 0);
}

#[test]
fn session_clock_keeps_event_order_when_new_input_moves_the_origin() {
    let (start, mut trace) = traced();
    trace.voice(start);
    let old = trace
        .boundary(Boundary::SocketReplaced, ms(start, 5_000))
        .unwrap();
    trace.voice(ms(start, 6_000));
    let audio = trace.audio(ms(start, 7_000)).unwrap();
    assert_eq!(old["elapsed_ms"], 5_000);
    assert_eq!(audio["elapsed_ms"], 1_000);
    assert_eq!(old["observed_ms"], 5_000);
    assert_eq!(audio["observed_ms"], 7_000);
}

#[test]
fn a_new_origin_and_a_pending_origin_keep_their_actual_sockets() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.set_socket(2);
    trace.voice(ms(start, 1_000));
    let record = trace.audio(ms(start, 1_500)).unwrap();
    assert_eq!(record["origin_socket"], 2);
    trace.prompt(ms(start, 1_600));
    trace.set_socket(3);
    trace.boundary(Boundary::SocketReplaced, ms(start, 2_000));
    let record = trace.audio(ms(start, 2_500)).unwrap();
    assert_eq!(record["origin"], "prompt");
    assert_eq!(record["origin_socket"], 2);
    assert_eq!(record["socket"], 3);
}

#[test]
fn local_cuts_discard_pending_inputs_instead_of_timing_the_hold() {
    for boundary in [Boundary::Held, Boundary::Paused] {
        let (start, mut trace) = traced();
        trace.voice(start);
        trace.audio(ms(start, 100));
        trace.voice(ms(start, 200));
        trace.voice_end(ms(start, 300));
        trace.boundary(boundary, ms(start, 400));
        trace.voice(ms(start, 45_000));
        let record = trace.audio(ms(start, 45_500)).unwrap();
        assert_eq!(record["elapsed_ms"], 500);
        assert_eq!(record["provider_voice_end_ms"], Value::Null);
    }
}

#[test]
fn a_late_end_belongs_to_speech_not_the_queued_platform_prompt() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.output(ms(start, 100), false);
    trace.prompt(ms(start, 200));
    trace.voice_end(ms(start, 300));
    let first = trace.audio(ms(start, 400)).unwrap();
    assert_eq!(first["provider_voice_end_ms"], 300);
    trace.voice_start();
    trace.voice(ms(start, 450));
    trace.boundary(Boundary::TurnComplete, ms(start, 500));
    let next = trace.audio(ms(start, 900)).unwrap();
    assert_eq!(next["origin"], "last_voiced_pcm");
    assert_eq!(next["elapsed_ms"], 450);
}

#[test]
fn a_platform_prompt_clears_milestones_of_superseded_speech() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.transcript(ms(start, 10));
    trace.voice_end(ms(start, 20));
    trace.prompt(ms(start, 30));
    let record = trace.audio(ms(start, 100)).unwrap();
    assert_eq!(record["origin"], "prompt");
    assert_eq!(record["provider_voice_end_ms"], Value::Null);
    assert_eq!(record["latest_transcript_ms"], Value::Null);
}

#[test]
fn replacement_after_a_tool_decision_keeps_the_unheard_reply_clock() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.output(ms(start, 1_000), true);
    trace.tool_response(ms(start, 1_100), 100, true);
    trace.set_socket(2);
    let replaced = trace
        .boundary(Boundary::SocketReplaced, ms(start, 5_000))
        .unwrap();
    assert_eq!(replaced["continuing"], true);
    trace.recovery_prompt(ms(start, 5_100));
    let record = trace.audio(ms(start, 6_000)).unwrap();
    assert_eq!(record["origin"], "last_voiced_pcm");
    assert_eq!(record["elapsed_ms"], 6_000);
    assert_eq!(record["turn"], 1);
}

#[test]
fn a_queued_prompt_does_not_turn_post_end_noise_into_new_speech() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.audio(ms(start, 100));
    trace.voice(ms(start, 200));
    trace.voice_end(ms(start, 250));
    trace.prompt(ms(start, 300));
    trace.voice(ms(start, 400));
    trace.boundary(Boundary::TurnComplete, ms(start, 500));
    trace.voice(ms(start, 600));
    let record = trace.audio(ms(start, 800)).unwrap();
    assert_eq!(record["origin"], "prompt");
    assert_eq!(record["elapsed_ms"], 500);
    assert_eq!(record["provider_voice_end_ms"], Value::Null);
}

#[test]
fn a_yield_is_stamped_only_while_the_reply_is_still_owed() {
    let (start, mut trace) = traced();
    trace.yielded(start);
    trace.voice_start();
    trace.voice(start);
    trace.yielded(ms(start, 600));
    trace.output(ms(start, 900), false);
    trace.yielded(ms(start, 950));
    let record = trace.audio(ms(start, 1_000)).unwrap();
    assert_eq!(record["candidate_yield_ms"], 600);
    trace.boundary(Boundary::TurnComplete, ms(start, 2_000));

    trace.voice(ms(start, 3_000));
    trace.yielded(ms(start, 3_200));
    trace.prompt(ms(start, 3_300));
    let record = trace.audio(ms(start, 3_800)).unwrap();
    assert_eq!(record["candidate_yield_ms"], Value::Null);
}

#[test]
fn an_unsent_tool_batch_does_not_start_the_continuation() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.output(ms(start, 1_000), true);
    trace.tool_response(ms(start, 1_100), 10, true);
    let failed = trace.tool_response(ms(start, 1_500), 10, false);
    assert_eq!(failed["last_tool_response_ms"], 1_100);
    assert_eq!(failed["tool_batches"], 1);
    assert_eq!(failed["tool_handler_ms"], 10);
}

#[test]
fn the_first_transcript_fragment_is_kept_while_the_latest_moves() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.transcript(ms(start, 200));
    trace.transcript(ms(start, 900));
    trace.output(ms(start, 1_000), false);
    let record = trace.audio(ms(start, 1_100)).unwrap();
    assert_eq!(record["first_transcript_ms"], 200);
    assert_eq!(record["latest_transcript_ms"], 900);
}

#[test]
fn a_new_utterance_restarts_its_first_transcript() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.transcript(ms(start, 100));
    trace.voice_start();
    trace.voice(ms(start, 60_000));
    trace.transcript(ms(start, 60_200));
    trace.voice_end(ms(start, 61_000));
    trace.output(ms(start, 61_000), false);
    let record = trace.audio(ms(start, 61_100)).unwrap();
    assert_eq!(record["first_transcript_ms"], 200);
}

#[test]
fn a_yield_before_a_new_utterance_does_not_end_it() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.yielded(ms(start, 500));
    trace.voice_start();
    trace.voice(ms(start, 40_000));
    trace.output(ms(start, 43_000), false);
    let record = trace.audio(ms(start, 43_100)).unwrap();
    assert_eq!(record["candidate_yield_ms"], Value::Null);
}

#[test]
fn records_carry_the_last_single_generation_context_size() {
    let (start, mut trace) = traced();
    trace.observe_usage(5_000, 0);
    trace.observe_usage(9_000, 1_200);
    trace.observe_usage(0, 0);
    trace.voice(start);
    let record = trace.audio(ms(start, 700)).unwrap();
    assert_eq!(record["context_tokens"], 5_000);
}

#[test]
fn pending_candidate_yield_survives_generation_boundaries() {
    for boundary in [
        Boundary::TurnComplete,
        Boundary::Interrupted,
        Boundary::SocketReplaced,
    ] {
        let (start, mut trace) = traced();
        trace.voice(start);
        trace.audio(ms(start, 100));
        trace.voice_start();
        trace.voice(ms(start, 200));
        trace.yielded(ms(start, 400));
        trace.voice(ms(start, 450));
        trace.yielded(ms(start, 480));
        trace.voice_end(ms(start, 500));
        trace.boundary(boundary, ms(start, 600));
        let record = trace.audio(ms(start, 900)).unwrap();
        assert_eq!(record["candidate_yield_ms"], 280);
        assert_eq!(record["provider_voice_end_ms"], 300);
    }
}

#[test]
fn pending_yield_does_not_survive_a_new_utterance_or_prompt() {
    for prompt in [false, true] {
        let (start, mut trace) = traced();
        trace.voice(start);
        trace.audio(ms(start, 100));
        trace.voice_start();
        trace.voice(ms(start, 200));
        trace.yielded(ms(start, 400));
        if prompt {
            trace.prompt(ms(start, 500));
        } else {
            trace.voice_start();
            trace.voice(ms(start, 500));
        }
        trace.boundary(Boundary::TurnComplete, ms(start, 600));
        assert_eq!(
            trace.audio(ms(start, 900)).unwrap()["candidate_yield_ms"],
            Value::Null
        );
    }
}

#[test]
fn local_cuts_discard_pending_candidate_yields() {
    for boundary in [Boundary::Paused, Boundary::Held] {
        let (start, mut trace) = traced();
        trace.voice(start);
        trace.audio(ms(start, 100));
        trace.voice_start();
        trace.voice(ms(start, 200));
        trace.yielded(ms(start, 400));
        trace.boundary(boundary, ms(start, 500));
        trace.voice(ms(start, 600));
        let record = trace.audio(ms(start, 900)).unwrap();
        assert_eq!(record["candidate_yield_ms"], Value::Null);
        assert_eq!(record["elapsed_ms"], 300);
    }
}

#[test]
fn queued_provider_milestones_use_the_last_input_before_receipt() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.voice(ms(start, 3120));
    trace.voice_end(ms(start, 3100));
    trace.output(ms(start, 3100), false);
    let record = trace.audio(ms(start, 3300)).unwrap();
    assert_eq!(record["elapsed_ms"], 3300);
    assert_eq!(record["provider_voice_end_ms"], 3100);
    assert_eq!(record["first_provider_output_ms"], 3100);
}

#[test]
fn a_queued_silent_completion_cannot_close_newer_input() {
    let (start, mut trace) = traced();
    trace.prompt(start);
    trace.voice(ms(start, 910));
    assert!(
        trace
            .boundary(Boundary::TurnComplete, ms(start, 900))
            .is_none()
    );
    let record = trace.audio(ms(start, 1500)).unwrap();
    assert_eq!(record["origin"], "last_voiced_pcm");
    assert_eq!(record["elapsed_ms"], 590);
}

#[test]
fn queued_end_rolls_pending_speech_back_to_its_receipt_boundary() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.audio(ms(start, 100));
    trace.voice_start();
    trace.voice(ms(start, 200));
    trace.voice(ms(start, 400));
    trace.voice_end(ms(start, 300));
    trace.boundary(Boundary::TurnComplete, ms(start, 500));
    let record = trace.audio(ms(start, 900)).unwrap();
    assert_eq!(record["elapsed_ms"], 700);
    assert_eq!(record["provider_voice_end_ms"], 100);
}

#[test]
fn yielding_freezes_the_current_origin_until_speech_restarts() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.yielded(ms(start, 100));
    trace.voice(ms(start, 200));
    let record = trace.audio(ms(start, 500)).unwrap();
    assert_eq!(record["candidate_yield_ms"], 100);
    assert_eq!(record["elapsed_ms"], 500);
}

#[test]
fn output_received_before_a_prompt_preserves_that_prompt_for_the_next_reply() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.prompt(ms(start, 30));
    trace.output(ms(start, 20), false);
    assert_eq!(
        trace.audio(ms(start, 40)).unwrap()["origin"],
        "last_voiced_pcm"
    );
    trace.boundary(Boundary::TurnComplete, ms(start, 50));
    let record = trace.audio(ms(start, 100)).unwrap();
    assert_eq!(record["origin"], "prompt");
    assert_eq!(record["elapsed_ms"], 70);
}

#[test]
fn queued_end_preserves_a_prompt_that_arrived_after_receipt() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.prompt(ms(start, 30));
    trace.voice_end(ms(start, 20));
    let record = trace.audio(ms(start, 100)).unwrap();
    assert_eq!(record["origin"], "prompt");
    assert_eq!(record["elapsed_ms"], 70);
    assert_eq!(record["provider_voice_end_ms"], Value::Null);
}

#[test]
fn queued_end_restores_a_pending_prompt_over_trailing_microphone_frames() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.audio(ms(start, 100));
    trace.prompt(ms(start, 200));
    trace.voice(ms(start, 300));
    trace.voice_end(ms(start, 290));
    trace.boundary(Boundary::TurnComplete, ms(start, 400));
    let record = trace.audio(ms(start, 500)).unwrap();
    assert_eq!(record["origin"], "prompt");
    assert_eq!(record["elapsed_ms"], 300);
}

#[test]
fn output_received_before_any_input_has_a_provider_origin_and_preserves_the_input() {
    let (start, mut trace) = traced();
    trace.voice(ms(start, 300));
    trace.output(ms(start, 290), false);
    let record = trace.audio(ms(start, 306)).unwrap();
    assert_eq!(record["origin"], "provider_event");
    assert_eq!(record["first_provider_output_ms"], 0);
    assert_eq!(record["elapsed_ms"], 16);
    trace.boundary(Boundary::TurnComplete, ms(start, 295));
    let record = trace.audio(ms(start, 600)).unwrap();
    assert_eq!(record["origin"], "last_voiced_pcm");
    assert_eq!(record["elapsed_ms"], 300);
}

#[test]
fn pending_speech_transcripts_survive_either_generation_ending() {
    for boundary in [Boundary::TurnComplete, Boundary::Interrupted] {
        let (start, mut trace) = traced();
        trace.voice(start);
        trace.audio(ms(start, 50));
        trace.voice_start();
        trace.voice(ms(start, 100));
        trace.transcript(ms(start, 200));
        trace.audio(ms(start, 225));
        trace.voice(ms(start, 250));
        trace.transcript(ms(start, 300));
        trace.voice_end(ms(start, 450));
        trace.boundary(boundary, ms(start, 500));
        let record = trace.audio(ms(start, 900)).unwrap();
        assert_eq!(record["first_transcript_ms"], -50);
        assert_eq!(record["latest_transcript_ms"], 50);
        assert_eq!(record["provider_voice_end_ms"], 200);
    }
}

#[test]
fn end_received_before_all_pending_frames_drops_the_spurious_reply() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.output(ms(start, 500), false);
    trace.voice(ms(start, 600));
    trace.voice_end(ms(start, 590));
    trace.boundary(Boundary::TurnComplete, ms(start, 1000));
    let record = trace.audio(ms(start, 30_000)).unwrap();
    assert_eq!(record["origin"], "provider_event");
    assert_eq!(record["elapsed_ms"], 0);
}

#[test]
fn end_received_before_all_local_frames_uses_the_provider_boundary() {
    let (start, mut trace) = traced();
    trace.voice(ms(start, 600));
    trace.voice_end(ms(start, 590));
    let record = trace.audio(ms(start, 1000)).unwrap();
    assert_eq!(record["origin"], "provider_voice_end");
    assert_eq!(record["elapsed_ms"], 410);
    assert_eq!(record["provider_voice_end_ms"], 0);
}

#[test]
fn origin_correction_restores_speech_milestones_superseded_by_a_prompt() {
    for pending in [false, true] {
        let (start, mut trace) = traced();
        if pending {
            trace.prompt(start);
            trace.audio(ms(start, 50));
        }
        trace.voice_start();
        trace.voice(ms(start, 100));
        trace.transcript(ms(start, 200));
        trace.yielded(ms(start, 800));
        trace.voice_end(ms(start, 1200));
        trace.prompt(ms(start, 1300));
        if pending {
            trace.boundary(Boundary::TurnComplete, ms(start, 1400));
        }
        trace.output(ms(start, 1250), false);
        let record = trace.audio(ms(start, 1500)).unwrap();
        assert_eq!(record["origin"], "last_voiced_pcm");
        assert_eq!(record["provider_voice_end_ms"], 1100);
        assert_eq!(record["candidate_yield_ms"], 700);
        assert_eq!(record["first_transcript_ms"], 100);
        trace.boundary(Boundary::TurnComplete, ms(start, 1600));
        let record = trace.audio(ms(start, 1800)).unwrap();
        assert_eq!(record["origin"], "prompt");
        assert_eq!(record["candidate_yield_ms"], Value::Null);
    }
}

#[test]
fn queued_output_does_not_revoke_transcripts_for_the_same_pending_utterance() {
    let (start, mut trace) = traced();
    trace.prompt(start);
    trace.voice_start();
    trace.voice(ms(start, 50));
    trace.voice(ms(start, 60));
    trace.output(ms(start, 40), false);
    trace.voice(ms(start, 70));
    trace.transcript(ms(start, 90));
    trace.boundary(Boundary::Interrupted, ms(start, 100));
    let record = trace.audio(ms(start, 150)).unwrap();
    assert_eq!(record["first_transcript_ms"], 20);
}

#[test]
fn provider_milestones_queued_before_a_prompt_attach_to_the_original_speech() {
    for pending in [false, true] {
        let (start, mut trace) = traced();
        if pending {
            trace.prompt(start);
            trace.audio(ms(start, 50));
        }
        trace.voice_start();
        trace.voice(ms(start, 100));
        trace.voice(ms(start, 105));
        trace.prompt(ms(start, 130));
        trace.transcript(ms(start, 110));
        trace.voice_end(ms(start, 115));
        if pending {
            trace.boundary(Boundary::TurnComplete, ms(start, 150));
        }
        trace.output(ms(start, 120), false);
        let record = trace.audio(ms(start, 200)).unwrap();
        assert_eq!(record["origin"], "last_voiced_pcm");
        assert_eq!(record["provider_voice_end_ms"], 10);
        assert_eq!(record["first_transcript_ms"], 5);
    }
}

#[test]
fn a_rollback_to_an_earlier_frame_restores_the_utterances_transcript() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.voice(ms(start, 10));
    trace.transcript(ms(start, 2));
    trace.prompt(ms(start, 30));
    trace.output(ms(start, 5), false);
    let record = trace.audio(ms(start, 40)).unwrap();
    assert_eq!(record["first_transcript_ms"], 2);
    assert_eq!(record["latest_transcript_ms"], 2);
}

#[test]
fn yielding_before_a_release_prompt_freezes_its_origin_until_speech_restarts() {
    let (start, mut trace) = traced();
    trace.yielded(start);
    trace.prompt(ms(start, 100));
    trace.voice(ms(start, 200));
    let record = trace.audio(ms(start, 600)).unwrap();
    assert_eq!(record["origin"], "prompt");
    assert_eq!(record["elapsed_ms"], 500);
    trace.boundary(Boundary::TurnComplete, ms(start, 700));
    trace.voice_start();
    trace.voice(ms(start, 900));
    assert_eq!(
        trace.audio(ms(start, 1000)).unwrap()["origin"],
        "last_voiced_pcm"
    );
}

#[test]
fn local_cuts_reset_a_yield_even_when_no_trace_was_opened() {
    for boundary in [Boundary::Held, Boundary::Paused] {
        let (start, mut trace) = traced();
        trace.yielded(start);
        assert!(trace.boundary(boundary, ms(start, 100)).is_none());
        trace.voice(ms(start, 200));
        let record = trace.audio(ms(start, 500)).unwrap();
        assert_eq!(record["origin"], "last_voiced_pcm");
        assert_eq!(record["elapsed_ms"], 300);
    }
}

/// A long utterance holds one input per voiced frame only as far back as an
/// event could still need, and the latest frame is still the origin.
#[test]
fn a_long_utterance_keeps_a_bounded_input_history() {
    let start = Instant::now();
    let mut latency = ReplyLatency::default();
    for frame in 0..6_000 {
        latency.voice(start + Duration::from_millis(frame * 10));
    }
    let held = latency.trace.as_ref().unwrap().inputs.len();
    assert!(held <= INPUT_PRUNE_AT, "{held}");
    let last = start + Duration::from_millis(59_990);
    let record = latency.audio(last + Duration::from_millis(500)).unwrap();
    assert_eq!(record["origin"], "last_voiced_pcm");
    assert_eq!(record["elapsed_ms"], 500);
}

/// Pruning keeps every voiced frame inside the horizon, the newest one past
/// it, and every platform prompt, so an event still in flight finds the input
/// it answers and a prompt is never lost to a long utterance.
#[test]
fn pruning_keeps_the_horizon_the_newest_stale_frame_and_every_prompt() {
    let start = Instant::now();
    let mut inputs = vec![(Origin::Prompt, start, 1)];
    for frame in 1..INPUT_PRUNE_AT as u64 {
        push_input(&mut inputs, (Origin::Voice, ms(start, frame * 10), 1));
    }
    assert_eq!(inputs.len(), INPUT_PRUNE_AT);
    push_input(&mut inputs, (Origin::Voice, ms(start, 20_480), 1));
    let kept: Vec<(Origin, u128)> = inputs
        .iter()
        .map(|(origin, at, _)| (*origin, at.duration_since(start).as_millis()))
        .collect();
    assert_eq!(kept.len(), 1_003);
    assert_eq!(
        kept[..3],
        [
            (Origin::Prompt, 0),
            (Origin::Voice, 10_470),
            (Origin::Voice, 10_480)
        ]
    );
    assert_eq!(kept.last(), Some(&(Origin::Voice, 20_480)));
}

/// Consecutive voiced frames are one kind of input and share one snapshot of
/// the milestones; a snapshot per frame would grow with the utterance and
/// nothing prunes them.
#[test]
fn consecutive_voiced_frames_save_no_milestone_snapshots() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.transcript(ms(start, 5));
    for frame in 1..100 {
        trace.voice(ms(start, frame * 10));
    }
    assert!(trace.trace.as_ref().unwrap().saved.is_empty());
}

/// Speech after a platform prompt saves what the prompt had seen, so output
/// queued before that speech still reports the prompt's own transcript.
#[test]
fn rolling_back_to_a_prompt_restores_what_it_had_seen_before_speech() {
    let (start, mut trace) = traced();
    trace.prompt(start);
    trace.transcript(ms(start, 10));
    trace.voice(ms(start, 20));
    trace.transcript(ms(start, 25));
    trace.voice_start();
    trace.output(ms(start, 15), false);
    let record = trace.audio(ms(start, 40)).unwrap();
    assert_eq!(record["origin"], "prompt");
    assert_eq!(record["first_transcript_ms"], 10);
    assert_eq!(record["latest_transcript_ms"], 10);
}

/// Frames behind a started reply with no new start of speech are the tail of
/// the utterance it answers, so that utterance's transcription is not credited
/// to the next reply, whether the frames came before or after the output was
/// received.
#[test]
fn trailing_frames_without_a_new_start_do_not_claim_its_transcript() {
    for queued in [false, true] {
        let (start, mut trace) = traced();
        trace.voice(start);
        if queued {
            trace.voice(ms(start, 100));
            trace.output(ms(start, 50), false);
            trace.transcript(ms(start, 60));
        } else {
            trace.audio(ms(start, 50));
            trace.voice(ms(start, 100));
            trace.transcript(ms(start, 150));
        }
        trace.boundary(Boundary::TurnComplete, ms(start, 200));
        let record = trace.audio(ms(start, 300)).unwrap();
        assert_eq!(record["origin"], "last_voiced_pcm");
        assert_eq!(record["elapsed_ms"], 200);
        assert_eq!(record["first_transcript_ms"], Value::Null);
        assert_eq!(record["latest_transcript_ms"], Value::Null);
    }
}

/// A start of speech that arrives after its first frames were queued behind a
/// reply makes those frames an utterance whose transcription counts.
#[test]
fn a_late_start_of_speech_admits_the_queued_frames_transcript() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.audio(ms(start, 50));
    trace.voice(ms(start, 100));
    trace.voice_start();
    trace.transcript(ms(start, 200));
    trace.boundary(Boundary::TurnComplete, ms(start, 300));
    let record = trace.audio(ms(start, 500)).unwrap();
    assert_eq!(record["elapsed_ms"], 400);
    assert_eq!(record["first_transcript_ms"], 100);
    assert_eq!(record["latest_transcript_ms"], 100);
}

/// A new utterance the microphone never registered as voiced still has its own
/// end, and the reply after it must not report the previous utterance's words.
#[test]
fn a_new_utterance_without_a_voiced_frame_reports_no_earlier_transcript() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.transcript(ms(start, 10));
    trace.voice_start();
    trace.voice_end(ms(start, 200));
    let record = trace.audio(ms(start, 500)).unwrap();
    assert_eq!(record["provider_voice_end_ms"], 200);
    assert_eq!(record["first_transcript_ms"], Value::Null);
    assert_eq!(record["latest_transcript_ms"], Value::Null);
}

/// A yield the room loop handled after frames the queued output preceded ends
/// the wait for those frames' reply, not for the one already under way.
#[test]
fn a_yield_after_frames_behind_queued_output_goes_to_the_next_reply() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.voice(ms(start, 100));
    trace.yielded(ms(start, 150));
    trace.output(ms(start, 50), false);
    let first = trace.audio(ms(start, 60)).unwrap();
    assert_eq!(first["candidate_yield_ms"], Value::Null);
    trace.boundary(Boundary::TurnComplete, ms(start, 200));
    let next = trace.audio(ms(start, 300)).unwrap();
    assert_eq!(next["elapsed_ms"], 200);
    assert_eq!(next["candidate_yield_ms"], 50);
}

/// A transcript received no later than the first output, including one that
/// came in the same provider message, belongs to the reply that output began,
/// not to the frames queued behind it.
#[test]
fn transcripts_received_by_the_first_output_stay_with_that_reply() {
    for heard in [10, 15] {
        let (start, mut trace) = traced();
        trace.voice(start);
        trace.voice(ms(start, 20));
        trace.transcript(ms(start, heard));
        trace.output(ms(start, 15), false);
        let first = trace.audio(ms(start, 16)).unwrap();
        assert_eq!(first["first_transcript_ms"], heard);
        assert_eq!(first["latest_transcript_ms"], heard);
        trace.boundary(Boundary::TurnComplete, ms(start, 30));
        let next = trace.audio(ms(start, 100)).unwrap();
        assert_eq!(next["elapsed_ms"], 80);
        assert_eq!(next["first_transcript_ms"], Value::Null);
        assert_eq!(next["latest_transcript_ms"], Value::Null);
    }
}

/// An end that rolls queued speech back to an earlier frame keeps that
/// speech's transcript; only a platform prompt clears it.
#[test]
fn queued_end_keeps_the_transcript_of_the_speech_it_rolls_back_to() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.audio(ms(start, 100));
    trace.voice_start();
    trace.voice(ms(start, 200));
    trace.transcript(ms(start, 250));
    trace.voice(ms(start, 400));
    trace.voice_end(ms(start, 300));
    trace.boundary(Boundary::TurnComplete, ms(start, 500));
    let record = trace.audio(ms(start, 900)).unwrap();
    assert_eq!(record["elapsed_ms"], 700);
    assert_eq!(record["provider_voice_end_ms"], 100);
    assert_eq!(record["first_transcript_ms"], 50);
    assert_eq!(record["latest_transcript_ms"], 50);
}

/// A turn that completes in the same provider message as the transcript that
/// opened its trace is a silent completion of that trace, not a stale one.
#[test]
fn a_completion_in_the_transcripts_own_message_closes_its_trace() {
    let (start, mut trace) = traced();
    trace.transcript(start);
    let record = trace.boundary(Boundary::TurnComplete, start).unwrap();
    assert_eq!(record["origin"], "transcript");
    assert_eq!(
        trace.audio(ms(start, 100)).unwrap()["origin"],
        "provider_event"
    );
}

/// Only a socket replacement says whether the reply carries on; one that
/// comes after the reply was heard closes it, and other boundaries say nothing.
#[test]
fn only_a_socket_replacement_says_whether_the_reply_continues() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.audio(ms(start, 500));
    let replaced = trace
        .boundary(Boundary::SocketReplaced, ms(start, 1_000))
        .unwrap();
    assert_eq!(replaced["continuing"], false);
    trace.voice(ms(start, 2_000));
    trace.audio(ms(start, 2_500));
    let complete = trace
        .boundary(Boundary::TurnComplete, ms(start, 3_000))
        .unwrap();
    assert_eq!(complete["continuing"], Value::Null);
}

/// Output splits the milestones at its receipt: the trace keeps what was
/// received by then and the next reply takes the rest. A milestone exactly at
/// the split must land on one side only, the trace's, or it is either counted
/// twice or lost.
#[test]
fn a_milestone_at_the_split_instant_is_received_by_it() {
    let now = ms(Instant::now(), 1_000);
    assert!(received_by(now - Duration::from_millis(1), now));
    assert!(received_by(now, now));
    assert!(!received_by(now + Duration::from_millis(1), now));
}

/// Every milestone after an output's receipt moves to the reply queued behind
/// it, and none stays on the reply that output began: the split loses nothing
/// and counts nothing twice.
#[test]
fn milestones_after_the_output_receipt_move_whole_to_the_next_reply() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.voice(ms(start, 100));
    trace.transcript(ms(start, 120));
    trace.voice_end(ms(start, 150));
    trace.yielded(ms(start, 160));
    trace.output(ms(start, 50), false);
    let first = trace.audio(ms(start, 60)).unwrap();
    trace.boundary(Boundary::TurnComplete, ms(start, 200));
    let next = trace.audio(ms(start, 300)).unwrap();
    for (milestone, offset) in [
        ("provider_voice_end_ms", 50),
        ("candidate_yield_ms", 60),
        ("first_transcript_ms", 20),
        ("latest_transcript_ms", 20),
    ] {
        assert_eq!(first[milestone], Value::Null, "{milestone}");
        assert_eq!(next[milestone], offset, "{milestone}");
    }
}

/// An end at exactly the trace's start belongs to the trace, as the
/// `trace.start <= now` arm that stamps it decides, so the utterance a start
/// of speech superseded must not be given it too.
#[test]
fn an_end_at_the_origin_instant_is_not_also_given_to_the_superseded_utterance() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.voice_start();
    trace.voice_end(start);
    let current = trace.trace.as_ref().unwrap();
    assert_eq!(current.voice_end, Some(start));
    assert!(
        current
            .saved
            .iter()
            .all(|(_, saved)| saved.voice_end.is_none())
    );
}

/// A transcript at exactly the trace's start is the current utterance's, as
/// the arm that stamps the trace decides, not a historical one for the
/// utterance a start of speech superseded.
#[test]
fn a_transcript_at_the_origin_instant_belongs_to_the_current_utterance() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.voice_start();
    trace.transcript(start);
    let record = trace.audio(ms(start, 100)).unwrap();
    assert_eq!(record["first_transcript_ms"], 0);
    assert_eq!(record["latest_transcript_ms"], 0);
}

/// An end at exactly the queued input's instant is that input's own, stamped
/// on the pending reply, and is not also written into the utterance a start
/// of speech superseded.
#[test]
fn a_queued_end_at_the_pending_instant_is_not_given_to_the_superseded_utterance() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.audio(ms(start, 50));
    trace.voice(ms(start, 100));
    trace.voice_start();
    trace.voice_end(ms(start, 100));
    let pending = trace.pending.as_ref().unwrap();
    assert_eq!(pending.voice_end, Some(ms(start, 100)));
    assert!(
        pending
            .saved
            .iter()
            .all(|(_, saved)| saved.voice_end.is_none())
    );
}

/// A transcript at exactly the queued input's instant is that input's own,
/// just as one at the trace's start is the trace's, so the next reply reports
/// it rather than a superseded utterance keeping it.
#[test]
fn a_transcript_at_the_pending_instant_belongs_to_the_queued_utterance() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.audio(ms(start, 50));
    trace.voice_start();
    trace.voice(ms(start, 100));
    trace.voice_start();
    trace.transcript(ms(start, 100));
    trace.boundary(Boundary::TurnComplete, ms(start, 200));
    let record = trace.audio(ms(start, 300)).unwrap();
    assert_eq!(record["first_transcript_ms"], 0);
    assert_eq!(record["latest_transcript_ms"], 0);
}

/// An end at exactly the queued input's instant leaves nothing received after
/// it to roll back, so the queued reply loses none of its milestones.
#[test]
fn a_queued_end_at_the_pending_instant_rolls_nothing_back() {
    let (start, mut trace) = traced();
    trace.voice(start);
    trace.audio(ms(start, 50));
    trace.voice(ms(start, 100));
    trace.voice_start();
    trace.yielded(ms(start, 105));
    trace.voice_end(ms(start, 100));
    trace.boundary(Boundary::TurnComplete, ms(start, 200));
    let record = trace.audio(ms(start, 300)).unwrap();
    assert_eq!(record["provider_voice_end_ms"], 0);
    assert_eq!(record["candidate_yield_ms"], 5);
}
