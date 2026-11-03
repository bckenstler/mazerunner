"""The promo reel's soundtrack: a synthwave cue rendered from code.

Nothing is sampled or downloaded. Every voice is built from sine partials,
filtered noise and envelopes, so the track is reproducible and carries no
licence beyond the repository's own. 100 BPM, A minor, the i–VI–III–VII loop
(Am F C G) that half of the genre runs on:

  intro    pad + echoing arpeggio, filter opening    (title card)
  build    + driving eighth-note bass and kick       (first clips)
  full     + gated-reverb snare and hats             (the rest of the clips)
  hook     + lead melody                             (the data scenes)
  outro    everything fades on the last bar

Usage: uv run python scripts/make_music.py [seconds] [out.wav]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, sosfilt

SR = 44100
BPM = 100
BEAT = 60 / BPM
BAR = 4 * BEAT
RNG = np.random.default_rng(1985)

# Am, F, C, G as (bass root, chord tones) in MIDI note numbers.
PROGRESSION = [
    (45, (57, 60, 64)),   # Am: A C E
    (41, (57, 60, 65)),   # F:  A C F (voiced close to Am)
    (48, (55, 60, 64)),   # C:  G C E
    (43, (55, 59, 62)),   # G:  G B D
]
# The hook, one bar per chord: (sixteenth offset, MIDI note, length in sixteenths).
HOOK = [
    [(0, 76, 3), (4, 74, 2), (6, 72, 2), (10, 69, 2), (12, 72, 4)],
    [(0, 69, 3), (4, 72, 2), (6, 74, 2), (8, 72, 2), (10, 69, 6)],
    [(0, 67, 3), (4, 72, 2), (6, 76, 4), (10, 74, 2), (12, 72, 4)],
    [(0, 74, 3), (4, 71, 2), (6, 67, 4), (10, 69, 6)],
]


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def t_axis(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SR)) / SR


def saw(freq: float, seconds: float, detune_cents=(0.0,), partials: int = 40) -> np.ndarray:
    """Band-limited sawtooth from sine partials (no aliasing), optionally
    several detuned copies for width."""
    t = t_axis(seconds)
    out = np.zeros_like(t)
    for cents in detune_cents:
        f = freq * 2 ** (cents / 1200)
        phase = RNG.uniform(0, 2 * np.pi)
        for k in range(1, partials + 1):
            if k * f > SR / 2.2:
                break
            out += np.sin(2 * np.pi * k * f * t + phase * k) / k
    return out / len(detune_cents)


def square(freq: float, seconds: float, partials: int = 25) -> np.ndarray:
    t = t_axis(seconds)
    out = np.zeros_like(t)
    for k in range(1, 2 * partials, 2):
        if k * freq > SR / 2.2:
            break
        out += np.sin(2 * np.pi * k * freq * t) / k
    return out


def adsr(n: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    a_n, d_n, r_n = int(a * SR), int(d * SR), int(r * SR)
    env = np.full(n, s)
    env[:a_n] = np.linspace(0, 1, a_n, endpoint=False) if a_n else env[:a_n]
    env[a_n:a_n + d_n] = np.linspace(1, s, max(0, min(d_n, n - a_n)), endpoint=False)
    if r_n:
        env[-r_n:] *= np.linspace(1, 0, min(r_n, n))
    return env


def lowpass(x: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    return sosfilt(butter(order, min(cutoff, SR / 2.1), "low", fs=SR, output="sos"), x)


def highpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    return sosfilt(butter(2, cutoff, "high", fs=SR, output="sos"), x)


def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), x)


def place(track: np.ndarray, sound: np.ndarray, at: float, gain: float = 1.0) -> None:
    i = int(at * SR)
    if i >= len(track):
        return
    j = min(len(track), i + len(sound))
    track[i:j] += gain * sound[: j - i]


def echo(x: np.ndarray, delay: float, feedback: float, mix: float) -> np.ndarray:
    """Feedback delay line, the arpeggio's ping."""
    d = int(delay * SR)
    b = np.zeros(d + 1)
    b[0] = 1.0
    a = np.zeros(d + 1)
    a[0], a[d] = 1.0, -feedback
    wet = lfilter(b, a, np.concatenate([np.zeros(d), x]))[: len(x)]
    return x + mix * wet


# ------------------------------------------------------------------ voices

def kick() -> np.ndarray:
    t = t_axis(0.42)
    pitch = 45 + 95 * np.exp(-t * 28)
    body = np.sin(2 * np.pi * np.cumsum(pitch) / SR) * np.exp(-t * 7.5)
    click = highpass(RNG.standard_normal(len(t)), 3000) * np.exp(-t * 300) * 0.15
    return body + click


def snare() -> np.ndarray:
    """Noise and a 190 Hz body through a short, bright gated reverb: the
    1984 snare."""
    t = t_axis(0.5)
    noise = bandpass(RNG.standard_normal(len(t)), 900, 7500) * np.exp(-t * 14)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 22) * 0.6
    dry = noise + tone
    ir_t = t_axis(0.32)
    ir = RNG.standard_normal(len(ir_t)) * np.exp(-ir_t * 6)
    ir[int(0.30 * SR):] *= np.linspace(1, 0, len(ir) - int(0.30 * SR))   # the gate
    wet = np.convolve(dry, ir)[: len(dry)] * 0.035
    return dry * 0.8 + bandpass(wet, 400, 9000)


def hat(open_: bool = False) -> np.ndarray:
    t = t_axis(0.25 if open_ else 0.06)
    return highpass(RNG.standard_normal(len(t)), 7500) * np.exp(-t * (18 if open_ else 70))


def bass_note(midi: int, seconds: float) -> np.ndarray:
    x = saw(hz(midi), seconds, (0.0, 6.0), partials=30)
    x = lowpass(x, 750) + 0.6 * np.sin(2 * np.pi * hz(midi) * t_axis(seconds))
    return x * adsr(len(x), 0.004, 0.08, 0.7, 0.03)


def pad_chord(notes, seconds: float, cutoff: float) -> np.ndarray:
    x = sum(saw(hz(n), seconds, (-9.0, 0.0, 8.0), partials=24) for n in notes)
    x = lowpass(x, cutoff, order=2)
    return x * adsr(len(x), 0.35, 0.4, 0.8, 0.5)


def pluck(midi: int, seconds: float, cutoff: float) -> np.ndarray:
    x = 0.7 * square(hz(midi), seconds) + 0.5 * saw(hz(midi), seconds, (0.0, 7.0), partials=20)
    x = lowpass(x, cutoff)
    return x * adsr(len(x), 0.003, 0.12, 0.15, 0.05)


def lead_note(midi: int, seconds: float) -> np.ndarray:
    t = t_axis(seconds)
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.15) * 4, 0, 1)
    phase = 2 * np.pi * hz(midi) * np.cumsum(vib) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 18) if k * hz(midi) < SR / 2.2)
    x = lowpass(x, 3800)
    return x * adsr(len(x), 0.02, 0.1, 0.75, 0.12)


# -------------------------------------------------------------------- mix

def render(seconds: float) -> np.ndarray:
    """A stereo cue exactly `seconds` long."""
    n = int(seconds * SR)
    bars = int(np.ceil(seconds / BAR))
    six = BEAT / 4
    pad, arp, bass, drums, lead = (np.zeros(n) for _ in range(5))
    duck = np.ones(n)   # sidechain: the pad and bass breathe with the kick

    for b in range(bars):
        start = b * BAR
        root, chord = PROGRESSION[b % 4]
        # The filter opens over the intro, so the track rises with the title.
        open_ = min(1.0, (b + 1) / 4)
        place(pad, pad_chord(chord, BAR + 0.5, 900 + 2400 * open_), start, 0.16)

        tones = list(chord) + [chord[0] + 12]
        for k in range(16):
            note = tones[[0, 1, 2, 3, 2, 1, 0, 1][k % 8]] + 12
            place(arp, pluck(note, six * 1.6, 1500 + 3500 * open_), start + k * six, 0.10)

        if b >= 4:
            for k in range(8):
                place(bass, bass_note(root + (12 if k % 2 else 0), BEAT / 2 * 0.92), start + k * BEAT / 2, 0.30)
            for k in range(4):
                place(drums, kick(), start + k * BEAT, 0.55)
                i = int((start + k * BEAT) * SR)
                dip = 1 - 0.45 * np.exp(-np.arange(int(0.3 * SR)) / SR * 12)
                j = min(n, i + len(dip))
                duck[i:j] = np.minimum(duck[i:j], dip[: j - i])
        if b >= 8:
            for k in (1, 3):
                place(drums, snare(), start + k * BEAT, 0.42)
            for k in range(8):
                place(drums, hat(open_=k % 2 == 1), start + k * BEAT / 2 + BEAT / 4, 0.05 if k % 2 else 0.035)
        if 12 <= b < bars - 1:
            for off, note, length in HOOK[b % 4]:
                place(lead, lead_note(note, length * six), start + off * six, 0.16)

    arp = echo(arp, 3 * BEAT / 4, 0.38, 0.55)
    lead = echo(lead, 3 * BEAT / 4, 0.25, 0.35)
    left = pad * duck + bass * duck + drums + arp * 0.8 + lead * 1.0
    right = pad * duck + bass * duck + drums + np.roll(arp, int(0.011 * SR)) * 1.0 + np.roll(lead, int(0.007 * SR))
    mix = np.stack([left, right], axis=1)
    mix = np.tanh(mix * 1.3) / np.tanh(1.3)            # gentle tape-style glue
    fade_in, fade_out = int(1.0 * SR), int(min(4.0, BAR * 1.5) * SR)
    mix[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    mix[-fade_out:] *= np.linspace(1, 0, fade_out)[:, None] ** 1.5
    peak = np.abs(mix).max()
    return (mix / peak * 10 ** (-1.0 / 20)).astype(np.float32)


def write(seconds: float, path: Path) -> Path:
    """Render the cue and save it as 16-bit stereo WAV."""
    audio = render(seconds)
    wavfile.write(path, SR, (audio * 32767).astype(np.int16))
    return path


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 70.0
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("promo-music.wav")
    write(secs, out)
    print(f"{out} ({secs:.1f}s)")
