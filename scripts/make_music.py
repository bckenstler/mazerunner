"""The promo reel's soundtrack: an 8-bit racing cue rendered from code.

Nothing is sampled or downloaded. The voices are the four channels of an old
home console, rebuilt in numpy: two pulse waves (a 25% lead and a 12.5%
arpeggio), a 4-bit stepped triangle on bass and a noise channel for drums,
plus a sine kick for weight on laptop speakers. The pulses are band-limited
(PolyBLEP), so the edges buzz the way the chips did without aliasing.

The cue follows the reel rather than a fixed clock. Three beeps count down
over the title card and the race starts (crash, drums, hook) on the first
clip. The tempo, about 150 BPM, is stretched so a whole number of 8-bar
phrases fits between the start and the first data scene, where the track
lifts a whole tone for the final lap. It ends on a held tonic hit over the
end card instead of fading:

  intro     arpeggio + bass, countdown beeps        (title card)
  A         the hook: a 3+3+2 repeated-note figure   (clips)
  A         again, with 16th hats and an octave double
  B         a longer-note answer, ending in a run up into A
  A         then a fill into the key change
  A (+2)    the hook a tone higher                   (data scenes)
  hit       tonic chord, crash, ring out             (end card)

Usage: uv run python scripts/make_music.py [seconds] [out.wav]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, sosfilt

SR = 44100
NOMINAL_BPM = 150
RNG = np.random.default_rng(1986)

# Bar specs: (bass root, chord tones, lead notes). Lead notes are
# (sixteenth offset, MIDI note, length in sixteenths), MIDI 69 = A4.
A_CHORDS = [
    (45, (57, 60, 64)),   # Am
    (41, (57, 60, 65)),   # F
    (43, (55, 59, 62)),   # G
    (40, (55, 59, 64)),   # Em
    (45, (57, 60, 64)),
    (41, (57, 60, 65)),
    (43, (55, 59, 62)),
    (45, (57, 60, 64)),   # Am, the phrase resolves
]
# Every bar of the hook opens on the same 3+3+2 figure (a repeated note,
# then a step down), then climbs; the rhythm is the part that sticks.
A_LEAD = [
    [(0, 81, 3), (3, 81, 3), (6, 79, 2), (8, 81, 2), (10, 84, 4), (14, 83, 2)],
    [(0, 81, 3), (3, 81, 3), (6, 79, 2), (8, 77, 2), (10, 76, 2), (12, 77, 4)],
    [(0, 79, 3), (3, 79, 3), (6, 77, 2), (8, 79, 2), (10, 83, 4), (14, 86, 2)],
    [(0, 83, 6), (6, 79, 2), (8, 76, 8)],
    [(0, 81, 3), (3, 81, 3), (6, 79, 2), (8, 81, 2), (10, 84, 4), (14, 83, 2)],
    [(0, 81, 3), (3, 81, 3), (6, 79, 2), (8, 77, 2), (10, 76, 2), (12, 77, 4)],
    [(0, 79, 3), (3, 79, 3), (6, 77, 2), (8, 79, 2), (10, 83, 4), (14, 86, 2)],
    [(0, 84, 3), (3, 83, 3), (6, 79, 2), (8, 81, 8)],
]
B_CHORDS = [
    (41, (57, 60, 65)),   # F
    (43, (55, 59, 62)),   # G
    (45, (57, 60, 64)),   # Am
    (45, (57, 60, 64)),
    (41, (57, 60, 65)),
    (43, (55, 59, 62)),
    (40, (56, 59, 64)),   # E major, the pull back home
    (40, (56, 59, 64)),
]
B_LEAD = [
    [(0, 81, 6), (6, 79, 2), (8, 81, 4), (12, 84, 4)],
    [(0, 83, 6), (6, 81, 2), (8, 79, 4), (12, 74, 4)],
    [(0, 76, 6), (6, 81, 2), (8, 84, 4), (12, 88, 4)],
    [(0, 86, 4), (4, 84, 4), (8, 81, 8)],
    [(0, 81, 6), (6, 79, 2), (8, 81, 4), (12, 84, 4)],
    [(0, 86, 6), (6, 84, 2), (8, 83, 4), (12, 79, 4)],
    [(0, 80, 6), (6, 83, 2), (8, 88, 8)],
    # Held B, then a harmonic-minor run up the octave into the hook.
    [(0, 83, 8)] + [(8 + i, n, 1) for i, n in enumerate((76, 77, 80, 81, 83, 84, 86, 88))],
]
INTRO_CHORDS = [(45, (57, 60, 64)), (41, (57, 60, 65)), (43, (55, 59, 62)), (40, (56, 59, 64))]


def hz(midi):
    return 440.0 * 2 ** ((np.asarray(midi, dtype=float) - 69) / 12)


def t_axis(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SR)) / SR


def _blep(phase: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """PolyBLEP residual: smooths one discontinuity per cycle at phase 0."""
    out = np.zeros_like(phase)
    m = phase < dt
    x = phase[m] / dt[m]
    out[m] = x + x - x * x - 1
    m = phase > 1 - dt
    x = (phase[m] - 1) / dt[m]
    out[m] = x * x + x + x + 1
    return out


def pulse(freq, seconds: float, duty: float) -> np.ndarray:
    """Band-limited pulse wave; `freq` may be a per-sample array (vibrato,
    chip-style chord arpeggios)."""
    n = int(seconds * SR)
    f = np.broadcast_to(np.asarray(freq, dtype=float), (n,))
    dt = f / SR
    phase = np.cumsum(dt) % 1.0
    y = np.where(phase < duty, 1.0, -1.0)
    y += _blep(phase, dt) - _blep((phase - duty) % 1.0, dt)
    return y - (2 * duty - 1)


def triangle(freq, seconds: float) -> np.ndarray:
    """The console's bass voice: a triangle quantised to 16 steps."""
    n = int(seconds * SR)
    f = np.broadcast_to(np.asarray(freq, dtype=float), (n,))
    phase = np.cumsum(f / SR) % 1.0
    tri = 4 * np.abs(phase - 0.5) - 1
    return np.round((tri + 1) * 7.5) / 7.5 - 1


def env(n: int, decay: float, sustain: float, release: float = 0.02) -> np.ndarray:
    """Instant attack, exponential fall to `sustain`, short release."""
    t = np.arange(n) / SR
    e = sustain + (1 - sustain) * np.exp(-t / decay)
    r = min(n, int(release * SR))
    if r:
        e[-r:] *= np.linspace(1, 0, r)
    e[: min(n, 40)] *= np.linspace(0, 1, min(n, 40))   # declick
    return e


def lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    return sosfilt(butter(2, cutoff, "low", fs=SR, output="sos"), x)


def highpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    return sosfilt(butter(2, cutoff, "high", fs=SR, output="sos"), x)


def place(track: np.ndarray, sound: np.ndarray, at: float, gain: float = 1.0) -> None:
    i = int(round(at * SR))
    if i < 0:
        sound, i = sound[-i:], 0
    if i >= len(track) or not len(sound):
        return
    j = min(len(track), i + len(sound))
    track[i:j] += gain * sound[: j - i]


def echo(x: np.ndarray, delay: float, feedback: float) -> np.ndarray:
    d = int(delay * SR)
    b = np.zeros(d + 1)
    b[d] = 1.0
    a = np.zeros(d + 1)
    a[0], a[d] = 1.0, -feedback
    return lfilter(b, a, x)


# ------------------------------------------------------------------ voices

def lead_note(midi: int, seconds: float, duty: float = 0.25) -> np.ndarray:
    t = t_axis(seconds)
    # Vibrato only on held notes, arriving after the attack.
    depth = 0.007 * np.clip((t - 0.14) * 6, 0, 1)
    x = pulse(hz(midi) * (1 + depth * np.sin(2 * np.pi * 6 * t)), seconds, duty)
    return lowpass(x, 7000) * env(len(t), 0.12, 0.7, 0.03)


def arp_note(midi: int, seconds: float) -> np.ndarray:
    x = pulse(hz(midi), seconds, 0.125)
    return x * env(len(x), 0.05, 0.35, 0.01)


def chord_stab(notes, seconds: float, rate: float = 0.024) -> np.ndarray:
    """The chip chord: one channel cycling the chord tones every `rate`
    seconds, too fast to hear as a melody."""
    t = t_axis(seconds)
    f = hz(np.asarray(notes)[(t // rate).astype(int) % len(notes)])
    x = pulse(f, seconds, 0.25)
    return lowpass(x, 6000) * env(len(t), 0.09, 0.25, 0.02)


def bass_note(midi: int, seconds: float) -> np.ndarray:
    # A quiet square an octave up gives the triangle a body small speakers
    # can reproduce.
    x = triangle(hz(midi), seconds) + 0.22 * lowpass(pulse(hz(midi + 12), seconds, 0.5), 1400)
    return x * env(len(x), 0.2, 0.8, 0.015)


def kick() -> np.ndarray:
    t = t_axis(0.26)
    f = 48 + 170 * np.exp(-t * 32)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 11)
    click = highpass(RNG.standard_normal(len(t)), 2500) * np.exp(-t * 400) * 0.3
    return body + click


def _crunch(n: int, hold: int) -> np.ndarray:
    """Noise held for `hold` samples at a time: the chip's coarser hiss."""
    return np.repeat(RNG.uniform(-1, 1, n // hold + 1), hold)[:n]


def snare() -> np.ndarray:
    t = t_axis(0.2)
    noise = highpass(_crunch(len(t), 2), 600) * np.exp(-t * 19)
    tone = triangle(180 + 70 * np.exp(-t * 40), 0.2) * np.exp(-t * 30) * 0.5
    return noise + tone


def hat(accent: bool = False) -> np.ndarray:
    t = t_axis(0.05)
    return highpass(RNG.uniform(-1, 1, len(t)), 8000) * np.exp(-t * (55 if accent else 95))


def crash() -> np.ndarray:
    t = t_axis(1.8)
    return highpass(_crunch(len(t), 1), 3500) * np.exp(-t * 2.4)


def beep(midi: int, seconds: float) -> np.ndarray:
    x = pulse(hz(midi), seconds, 0.5)
    return x * env(len(x), 1.0, 1.0, 0.03)


# -------------------------------------------------------------------- mix

def _arrangement(pre_bars: int, post_bars: int):
    """Bar list after the start: (section, bar index, transpose, flags)."""
    pre_cycle = [("A", 0), ("A", 1), ("B", 0), ("A", 2)]
    bars = []
    k = 0
    while len(bars) < pre_bars:
        sec, rep = pre_cycle[k % len(pre_cycle)]
        bars += [(sec, i, 0, {"rep": rep}) for i in range(8)]
        k += 1
    bars = bars[:pre_bars]
    # After the lift: the hook from the top, then as many of its closing bars
    # as fit, so the last bar before the final hit is always the resolution.
    tail = list(range(8)) * (1 + post_bars // 8)
    order = tail[:post_bars] if post_bars <= 8 else list(range(8)) + tail[8 - (post_bars - 8) % 8:][: post_bars - 8]
    bars += [("A", i, 2, {"rep": 1}) for i in order]
    return bars


def render(seconds: float, go: float = 5.5, lift: float | None = None) -> np.ndarray:
    """A stereo cue exactly `seconds` long.

    `go` is when the race starts (the first clip) and `lift` when the key
    change lands (the first data scene); without `lift` the cue never
    modulates. The final hit falls on the last bar line at least two seconds
    before the end.
    """
    span = (lift if lift is not None else seconds) - go
    nominal_bar = 4 * 60 / NOMINAL_BPM
    pre_bars = max(8, 8 * round(span / (8 * nominal_bar))) if lift is not None else 10**6
    bar = span / pre_bars if lift is not None else nominal_bar
    six = bar / 16
    final_bar = int((seconds - 2.0 - go) // bar)
    pre_bars = min(pre_bars, final_bar)
    bars = _arrangement(pre_bars, final_bar - pre_bars)

    n = int(seconds * SR)
    lead, dbl, arp, bass, drums, fx = (np.zeros(n) for _ in range(6))

    # Intro: the arpeggio and a held bass over the title, then the countdown.
    for b in range(-4, 0):
        root, chord = INTRO_CHORDS[b + 4]
        start = go + b * bar
        tones = list(chord) + [chord[0] + 12]
        for k in range(16):
            gain = 0.05 + 0.05 * (b + 4 + k / 16) / 4
            place(arp, arp_note(tones[[0, 1, 2, 3, 2, 1, 0, 1][k % 8]] + 12, six * 0.8), start + k * six, gain)
        for k in (0, 8):
            place(bass, bass_note(root, six * 7.5), start + k * six, 0.3)
    for beat in (6, 4, 2):
        place(fx, beep(69, 0.16), go - beat * bar / 4, 0.16)
    place(fx, beep(81, 0.45), go, 0.16)
    place(drums, crash(), go, 0.22)

    for idx, (sec, i, shift, flags) in enumerate(bars):
        start = go + idx * bar
        root, chord = (A_CHORDS if sec == "A" else B_CHORDS)[i]
        root, chord = root + shift, [c + shift for c in chord]
        notes = (A_LEAD if sec == "A" else B_LEAD)[i]
        duty = 0.25 if sec == "A" else 0.5
        for off, note, length in notes:
            x = lead_note(note + shift, length * six * 0.94, duty)
            place(lead, x, start + off * six, 0.22)
            if flags["rep"] and sec == "A":
                place(dbl, lead_note(note + shift + 12, length * six * 0.94, 0.125), start + off * six, 0.05)

        if sec == "A":
            tones = chord + [chord[0] + 12]
            for k in range(16):
                place(arp, arp_note(tones[[0, 1, 2, 3, 2, 1, 2, 3][k % 8]] + 12, six * 0.8), start + k * six, 0.07)
        else:
            for k in range(2, 16, 4):
                place(arp, chord_stab([c + 12 for c in chord], six * 1.6), start + k * six, 0.09)

        for k in range(8):
            place(bass, bass_note(root + (12 if k % 2 else 0), six * 1.8), start + k * 2 * six, 0.34)

        last_of_phrase = i == 7
        into_lift = idx == pre_bars - 1 and lift is not None
        for k in (0, 4, 8, 12):
            place(drums, kick(), start + k * six, 0.62)
        for k in (4, 12):
            place(drums, snare(), start + k * six, 0.36)
        dense = flags["rep"] or sec == "B"
        for k in range(16) if dense else range(2, 16, 4):
            place(drums, hat(accent=k % 4 == 2), start + k * six, 0.07 if k % 4 == 2 else 0.04)
        if into_lift:
            for k in range(8, 16):
                place(drums, snare(), start + k * six, 0.16 + 0.03 * (k - 8))
            sweep = t_axis(bar / 2)
            rise = pulse(hz(57 + 24 * sweep / sweep[-1]), bar / 2, 0.25)
            place(fx, rise * np.linspace(0, 1, len(rise)) ** 2, start + bar / 2, 0.06)
        elif last_of_phrase:
            for k in (13, 14, 15):
                place(drums, snare(), start + k * six, 0.24)
        if idx == pre_bars and lift is not None:
            place(drums, crash(), start, 0.22)
        if i == 0 and idx and not (idx == pre_bars and lift is not None):
            place(drums, crash(), start, 0.12)

    # The final hit: tonic in every voice, left to ring over the end card.
    hit = go + final_bar * bar
    shift = 2 if lift is not None else 0
    ring = seconds - hit
    place(lead, lead_note(81 + shift, ring, 0.25) * np.exp(-t_axis(ring) * 1.1), hit, 0.22)
    place(arp, chord_stab([69 + shift, 72 + shift, 76 + shift], ring, 0.03) * 2.5 * np.exp(-t_axis(ring) * 1.2), hit, 0.08)
    place(bass, bass_note(45 + shift, ring) * np.exp(-t_axis(ring) * 1.4), hit, 0.38)
    place(drums, kick(), hit, 0.7)
    place(drums, crash(), hit, 0.26)

    slap = echo(lead, 3 * six, 0.3)
    left = lead + 0.25 * slap + 0.8 * dbl + 1.1 * arp + bass + drums + fx
    right = lead + 0.45 * slap + 1.2 * dbl + 0.7 * arp + bass + drums + fx
    mix = np.stack([left, right], axis=1)
    mix = np.tanh(mix * 1.2) / np.tanh(1.2)
    fade_in, fade_out = int(0.3 * SR), int(0.5 * SR)
    mix[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    mix[-fade_out:] *= np.linspace(1, 0, fade_out)[:, None]
    # Square waves are dense: peaking at -1 dBFS measured -8.5 LUFS, far
    # louder than web video. -6.5 dBFS lands near the usual -14 LUFS.
    return (mix / np.abs(mix).max() * 10 ** (-6.5 / 20)).astype(np.float32)


def write(seconds: float, path: Path, go: float = 5.5, lift: float | None = None) -> Path:
    """Render the cue and save it as 16-bit stereo WAV."""
    audio = render(seconds, go, lift)
    wavfile.write(path, SR, (audio * 32767).astype(np.int16))
    return path


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 74.4
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("promo-music.wav")
    write(secs, out, go=5.5, lift=secs - 20.7)
    print(f"{out} ({secs:.1f}s)")
