# Reply latency observability

The Rust agent logs one `codetrial latency {JSON}` line at each point a reply
passes on the server. `python3 scripts/analyze-latency.py server.log` joins
them into a per-reply waterfall, a chronological timeline and a closing
`key=value` summary. `--summary` prints only the summary, `--json` prints
everything, and omitting the filename reads stdin. Log prefixes from journald
or containers are accepted. The run fails when no record is found; `--strict`
also fails on any rejected record or when no reply with a known input origin
reached native capture. Provider-only output remains visible but cannot establish
a healthy input-to-capture latency verdict. No
transcript, tool arguments, responses, credentials or candidate media enter
these records.

## One trace per reply

Each reply is a trace keyed by `(pid, room, session, turn)`. `socket` is where
the record was written and `origin_socket` where its input arrived, so a reply
that waits through a socket replacement stays one turn and its total includes
the stall. Recovery briefings preserve that origin; they do not restart the
wait. The origin is the input the reply answers:

| Origin | Meaning |
| --- | --- |
| `last_voiced_pcm` | The latest voiced 10 ms input frame the room loop consumed. An energy threshold, not an acoustic speech-end detector. |
| `prompt` | A platform prompt sent to the model. |
| `transcript` | A candidate transcript with no voiced frame before it. |
| `provider_voice_end` | Gemini's end of speech with no voiced frame before it. |
| `provider_event` | Output nothing local asked for. Reads near zero by construction and is kept out of percentiles. |

Until the reply's first output, the latest input wins: speech after an idle
prompt, or a prompt after a stray noise frame, moves the origin. Voiced frames
after Gemini's end of speech do not, until Gemini reports a new start. Input
that arrives once output has begun, whether barge-in speech or a prompt queued
behind the generation, waits for the next reply instead of moving or ending
this one, and survives either a normal completion or an interruption. Trailing
transcripts alone do not queue another reply, and the runtime
late-transcript grace also applies to tracing.

Generations a pause or a thinking hold has cut off locally are closed then, as
`paused` or `held`, discarding queued input from before that local cut. Their late provider events are not attributed to the reply
that follows. An `interrupted` before any output keeps the clock, because the
barge-in that caused it is the input still waiting for an answer.

## Stages

| Stage | Written when |
| --- | --- |
| `tool_response` | A tool batch's responses were sent, with its execution, board resend and send times and the declared names of the tools it answered. A batch that failed to send is recorded but does not move `last_tool_response_ms`. |
| `audio_accepted` | The first usable chunk was accepted for framing. Logged after the handler; `enqueued` says whether a complete frame entered the worker queue. |
| `native_audio_capture` | The worker finished LiveKit `capture_frame` for that chunk's first frame; `captured` says whether it succeeded. |
| `native_audio_cancelled` | That frame was cancelled before capture, including a chunk still waiting for enough PCM to form a frame. |
| `audio_enqueue_failed` | The first frame could not enter a full or closed worker queue. It is not a native-capture attempt or a successful latency sample. |
| `turn_complete`, `interrupted` | The provider ended the generation. |
| `paused`, `held` | A local cut ended it. |
| `socket_replaced` | The socket was replaced. A reply without accepted audio keeps waiting (`continuing=true`), including tool-only generations. |

## Waterfall

With `CODETRIAL_GEMINI_TOOL_ACK` set, a tool round that holds back the reply a
candidate is waiting for, and has heard nothing of, queues a short
acknowledgement generated once per interview in the session's voice and accent
(`gemini-3.1-flash-tts-preview`, on the report keys). It plays at most once per
thing the candidate said, never while they are speaking again, and never
before a goodbye. It is not the reply: the watchdog and this record still wait
for the reply's own audio. Its first accepted chunk cancels the remaining clip
and clears buffered filler before the reply plays. Audio already transmitted
cannot be recalled. Cancelled queue entries are skipped without capture;
`output_queue_frames` can include them until the worker drains them. The record
stamps it as `tool_ack_ms`, and the summary's `tool_ack_lead` is the reply's native
capture less that stamp: how much sooner the candidate heard something. An
interview whose clip is not ready yet, or failed to generate, plays nothing.

Offsets share the origin and use the agent's monotonic clock. A milestone seen
before an origin that later moved is negative, and one never observed is
`null`. `observed_ms` may be negative when provider receipt precedes the first
local input. It uses a fixed session clock to order events even when a
later input moves the reply origin. The analyzer partitions origin to first native capture into
consecutive segments, so they sum to `server_total_ms`; `unattributed` absorbs
per-field millisecond rounding.

| Segment | Span |
| --- | --- |
| `candidate_yield` | Origin to the candidate choosing "Your turn is done", when that came inside the silence window. The candidate's own time; it replaces `silence_window`. |
| `silence_window` | A baseline allowance for a `last_voiced_pcm` origin: configured `GEMINI_SILENCE_MS`, capped by observed END or output. This is not measured provider endpointing time. |
| `endpointing` | The rest of the wait for Gemini's end of speech: uplink, provider voice detection, and whatever Gemini does before it reports END. |
| `response_start` | End of speech, or origin without one, to the first provider output. |
| `tool_decision` | The same span when that output was a tool call. |
| `tool_rounds` | First tool call to the last tool response sent, including any further model rounds between batches. |
| `post_tool_response` | Last tool response to first audio accepted. |
| `output_to_audio` | First output (an output transcript) to first audio accepted. |
| `output_queue` | Audio accepted to the worker dequeuing its first frame. |
| `native_capture` | LiveKit `capture_frame` of that frame. |

Overlays are independent spans, not additional waterfall segments.
`tool_handler_ms` is the local share of `tool_rounds` (execution, board resend
and send);
`provider_event_queue_ms` is from receipt of the provider message, before
parsing, to the start of the audio handler. It can begin before the reply origin when the
room loop consumes later microphone frames before handling that message.
`context_tokens` is the prompt size of the last single-generation turn before
the reply, and a cold socket clears it until its own first report. The summary
bands `beyond_window` by it (`context_0k_8k` and up) and fits
`context_slope_ms_per_1k`, a least-squares slope in milliseconds per thousand
tokens fitted within each kind of reply, so a long interview shows whether
replies slow as the conversation grows rather than that tool replies, which cost
more, come later. `stale_drop_ms`, `enqueue_ms` and `handler_tail_ms` split the audio handler
around the enqueue, the last being what the handler does after it, the
agent-state update included; `output_queue_frames` and `playout_ahead_ms` say how much
audio was already queued ahead of the reply's first frame, which separates a
first frame waiting behind earlier audio from one waiting on the worker. Adding overlays to the waterfall double counts time.

Gemini can report END in the same message as its first output, so END alone
does not separate waiting from answering; `end_with_output` counts the replies
whose first output came within 10 ms of END, where the segment named
endpointing also holds the model's own start. The configured silence window is a
comparison allowance, not proof of provider time spent waiting: explicit
yielding can end input sooner, and inference can overlap voice detection.
A yield ends the audio stream, so Gemini answers without waiting out the
window, and the trace stamps it as `candidate_yield_ms`. `beyond_window` is
total time outside the allowance or the yield, not measured inference. `transcript_before_end`
reports how long before END the first input transcript of the utterance
arrived, in whole milliseconds; zero means the words came with END and
cannot inform an earlier end-of-turn decision, and a negative value means they
came after it. The
dominant segment excludes both, and a reply is slow when its time
beyond the allowance reaches `--slow-ms` (1,500 by default). A reply at or past
`--stall-ms` (5,000 by default) is a stall. The threshold is a flag, not a
diagnosis: it sits in the gap between ordinary slow starts and sockets that
stopped answering, and a reply past it is reported as a rate, `stalls`, and
left out of every distribution, where a few would set the mean and the tail;
`beyond_window_with_stalls` keeps them in for the tail they set, `worst` still
lists them, and a sample of nothing but stalls is `SLOW`. Statistics separate origins, count only captured
replies, and report turns without capture by their last audio stage and by
whether a tool round came first (`unmeasured_by_tool_tool`,
`unmeasured_by_tool_no_tool`), since uncaptured replies are mostly the slower
kind. `kind_<origin>_tool` and `kind_<origin>_no_tool` split `beyond_window` by
whether a tool round came before the reply's first audio; evidence recorded
after it began does not make a tool reply. Percentiles
interpolate between ranks, to a tenth of a millisecond, and a statistic over
fewer than 20 replies is marked `small_n`: its p95 then lies between its two
largest values. Compare matched yield settings and input recordings; this
budget comparison cannot substitute for browser-perceived E2E measurement.

## What is not measured

Gemini Live is one combined audio and model pipeline. These records cannot
separate its speech recognition, endpointing, inference and synthesis.
Native capture also does not prove the candidate heard anything: LiveKit
transport, browser jitter buffering, autoplay and the audio device on the way
out, and input transport before the agent on the way in, are outside this
record. A browser-perceived figure needs a matched audio measurement in the
browser.

## Validation

The credential-free gate covers the trace lifecycle, the waterfall and the
parser. For a live comparison, run the Rust voice and barge-in flows in
`scripts/browser-check.sh` against the same provider, region, input recording
and silence setting. `BROWSER_CHECK_CAPTURE` includes a `latencyLog` field with
the latency lines the harness saw; pipe it into the analyzer. The harness keeps
a bounded tail of agent output, so this is a smoke test, not an archive of a
long session.

Input audio is sent to Gemini in 20 ms batches, two of LiveKit's 10 ms frames,
so the last words wait at most one frame for a batch. Keep the default silence
window unless a matched comparison shows it no longer cuts candidates off:
`silence_window` is usually the largest segment, and the one most likely to be
traded against interrupting a candidate who paused to think.

## Design lineage

The breakdown follows the per-turn analysis in livekit-traces-analyzer, which
splits each reply into speech recognition, end-of-turn detection, the
tool-decision model call, tool execution and the model call after it, with the
remainder shown as overhead, and which surfaced a silent loss of spans when an
export file was named differently. CodeTrial keeps the tool split and the
residual, joins asynchronous records by identity rather than by neighboring
lines, and refuses an empty analysis. It uses milestones measured on its own
clock instead of estimating provider stages it cannot see.
