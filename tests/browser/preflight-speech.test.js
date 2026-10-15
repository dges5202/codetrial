// Browser playback cannot prove physical audibility through system effects.
import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_RUNTIME_CONFIG,
  FAKE_SPEECH_CAPTURE,
  startStaticServer,
} from "./source.js";

let browser;
let server;
let base;
before(async () => {
  try {
    const { chromium } = await import("playwright");
    browser = await chromium.launch({
      args: [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        FAKE_SPEECH_CAPTURE,
      ],
    });
  } catch (error) {
    if (process.env.CI) throw error;
    return;
  }
  ({ server, base } = await startStaticServer({
    runtimeConfig: DEFAULT_RUNTIME_CONFIG,
  }));
});
after(async () => {
  await browser?.close();
  await new Promise((closed) => (server ? server.close(closed) : closed()));
});

async function preflight(t) {
  const page = await browser.newPage();
  t.after(() => page.close());
  await page.addInitScript(() => {
    const capture = navigator.mediaDevices.getUserMedia.bind(
      navigator.mediaDevices,
    );
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      const stream = await capture(constraints);
      if (constraints.audio) window.testMicrophone = stream.getAudioTracks()[0];
      return stream;
    };
  });
  await page.goto(`${base}/interview.html?problem=chargeback-pair-match`);
  await page.waitForFunction(
    () => window.testMicrophone?.readyState === "live",
  );
  return page;
}

test("preflight speech replays during capture and still requires the candidate's confirmation", async (t) => {
  if (!browser) return t.skip("playwright chromium unavailable");
  const page = await preflight(t);
  const response = await page.request.get(`${base}/audio/test-voice.wav`);
  assert.equal(response.status(), 200);
  assert.match(response.headers()["content-type"], /^audio\/wav/);
  const signal = await page.evaluate(async () => {
    const response = await fetch("./audio/test-voice.wav");
    const buffer = await new OfflineAudioContext(1, 1, 48000).decodeAudioData(
      await response.arrayBuffer(),
    );
    let peak = 0;
    for (const value of buffer.getChannelData(0))
      peak = Math.max(peak, Math.abs(value));
    return { peak, duration: buffer.duration };
  });
  assert.ok(signal.peak > 0.01, "the shipped speech sample contains a signal");
  assert.ok(signal.duration > 1 && signal.duration < 10);
  for (let count = 0; count < 3; count++) {
    await page.locator("#audio-test-speech").click();
    await page.waitForFunction(
      () => document.querySelector("#audio-test-voice").ended,
    );
    assert.equal(
      await page.evaluate(() => window.testMicrophone.readyState),
      "live",
    );
    assert.equal(
      await page
        .locator("#audio-step-output")
        .evaluate((node) => node.classList.contains("done")),
      false,
    );
    assert.equal(await page.locator("#audio-check-join").isDisabled(), true);
  }
  await page.locator("#audio-heard").click();
  await page.waitForFunction(() =>
    document.querySelector("#audio-step-output").classList.contains("done"),
  );
});

test("preflight speech reports a rejected play and lets the candidate retry", async (t) => {
  if (!browser) return t.skip("playwright chromium unavailable");
  const page = await preflight(t);
  await page.evaluate(() => {
    document.querySelector("#audio-test-voice").play = () =>
      Promise.reject(new DOMException("Autoplay blocked", "NotAllowedError"));
  });
  await page.locator("#audio-test-speech").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#audio-check-status")
      .textContent.includes("Could not play"),
  );
  assert.equal(await page.locator("#audio-test-speech").isEnabled(), true);
  assert.equal(await page.locator("#audio-heard").textContent(), "I heard it");
  await page.evaluate(() => {
    delete document.querySelector("#audio-test-voice").play;
  });
  await page.locator("#audio-test-speech").click();
  await page.waitForFunction(
    () => document.querySelector("#audio-test-voice").ended,
  );
  assert.equal(
    await page.locator("#audio-test-voice").evaluate((audio) => audio.error),
    null,
  );
});

test("preflight retries a failed speech load without confirming output", async (t) => {
  if (!browser) return t.skip("playwright chromium unavailable");
  const page = await preflight(t);
  let requests = 0;
  await page.route("**/audio/test-voice.wav", (route) => {
    requests++;
    return requests === 1
      ? route.fulfill({ status: 404, body: "not found" })
      : route.continue();
  });
  await page.locator("#audio-test-speech").click();
  await page.waitForFunction(
    () => document.querySelector("#audio-test-voice").error?.code === 4,
  );
  await page.waitForFunction(() =>
    document
      .querySelector("#audio-check-status")
      .textContent.includes("Could not"),
  );
  assert.equal(await page.locator("#audio-test-speech").isEnabled(), true);
  assert.equal(requests, 1);
  await page.locator("#audio-test-speech").click();
  await page.waitForFunction(
    () => document.querySelector("#audio-test-voice").ended,
  );
  assert.ok(requests > 1, "retry fetches the speech asset again");
  assert.equal(
    await page.locator("#audio-test-voice").evaluate((audio) => audio.error),
    null,
  );
  assert.equal(
    await page
      .locator("#audio-step-output")
      .evaluate((node) => node.classList.contains("done")),
    false,
  );
});

test("switching to the tone cancels pending speech without reporting a playback failure", async (t) => {
  if (!browser) return t.skip("playwright chromium unavailable");
  const page = await preflight(t);
  let releaseSample;
  const heldSample = new Promise((resolve) => {
    releaseSample = resolve;
  });
  await page.route("**/audio/test-voice.wav", async (route) => {
    await heldSample;
    await route.continue();
  });
  try {
    await page.locator("#audio-test-speech").click();
    assert.equal(await page.locator("#audio-test-speech").isDisabled(), true);
    await page.locator("#audio-test-tone").click();
    await page.waitForFunction(
      () => !document.querySelector("#audio-test-speech").disabled,
    );
    assert.equal(
      await page.locator("#audio-test-voice").evaluate((audio) => audio.paused),
      true,
    );
    assert.doesNotMatch(
      await page.locator("#audio-check-status").textContent(),
      /Could not/,
    );
  } finally {
    releaseSample();
  }
});

test("leaving the preflight stops speech before navigation", async (t) => {
  if (!browser) return t.skip("playwright chromium unavailable");
  const page = await preflight(t);
  let releaseLobby;
  const heldLobby = new Promise((resolve) => {
    releaseLobby = resolve;
  });
  await page.route(`${base}/`, async (route) => {
    await heldLobby;
    await route.continue();
  });
  try {
    await page.locator("#audio-test-speech").click();
    await page.waitForFunction(
      () => !document.querySelector("#audio-test-voice").paused,
    );
    const stopped = await page.evaluate(() => {
      document.querySelector("#audio-check-leave").click();
      const audio = document.querySelector("#audio-test-voice");
      return { paused: audio.paused, time: audio.currentTime };
    });
    assert.deepEqual(stopped, { paused: true, time: 0 });
  } finally {
    releaseLobby();
  }
  await page.waitForURL(`${base}/`);
});

test("starting the interview stops preflight speech while keeping the microphone", async (t) => {
  if (!browser) return t.skip("playwright chromium unavailable");
  const page = await preflight(t);
  await page.route("**/api/token", (route) =>
    route.fulfill({ status: 503, body: "local test" }),
  );
  await page.locator("#camera-skip").click();
  await page.locator("#audio-heard").click();
  await page.waitForFunction(
    () => !document.querySelector("#audio-check-join").disabled,
  );
  await page.locator("#audio-test-speech").click();
  await page.waitForFunction(
    () => !document.querySelector("#audio-test-voice").paused,
  );
  const stopped = await page.evaluate(() => {
    document.querySelector("#audio-check-join").click();
    const audio = document.querySelector("#audio-test-voice");
    return {
      paused: audio.paused,
      time: audio.currentTime,
      microphone: window.testMicrophone.readyState,
    };
  });
  assert.deepEqual(stopped, { paused: true, time: 0, microphone: "live" });
  assert.equal(await page.locator("#audio-check").isHidden(), true);
});
