"""
synth_engine.py — shared real-time audio engine
================================================

Contains Step (immutable audio snapshot) and ContinuousSynth (queue-based
producer/consumer bridge between a UI thread and the sounddevice audio thread).

Import this module in a sonification script that needs real-time playback.
The app object passed to ContinuousSynth must expose:
    - names: list[str]
    - index: int
    - playing_index: int
    - length: int
    - root: tk.Tk
    - current_note_var: tk.StringVar
    - track_enabled: dict[str, tk.BooleanVar]
    - get_step(idx) -> tuple of (name, freq_hz, amplitude)
    - get_n_samples(idx) -> int
    - get_wave() -> str
"""

import queue
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from utils import (
    FS,
    STREAM_BLOCKSIZE,
    SOFT_LIMITER_DRIVE,
    SOFT_LIMITER_CEILING,
    PREFILL,
    soft_limiter,
)


@dataclass(
    frozen=True
)  # immutable (frozen) which means we cannot change its attributes after creation; dataclass allows us to define a simple reusable object without __init__
class Step:
    """
    Immutable snapshot of one sixteenth-note step.

    Frozen so the audio thread can read it without any locking —
    nothing can mutate it after it is placed on the queue.
    """

    samples: int  # number of audio samples this step lasts
    tracks: tuple  # tuple of (name, freq_hz, amplitude) per active track
    wave: str  # oscillator waveform name
    index: int  # data index this step was computed from (used to sync the plot)


class ContinuousSynth:
    """
    Real-time synthesiser bridging a UI thread and the sounddevice audio thread.

    Responsibilities:
    - Build Step objects from current app state (UI thread).
    - Keep a pre-filled SimpleQueue so the audio thread never starves.
    - Render PCM samples from queued Steps (audio thread).
    - Flush and refill the queue instantly when UI parameters change.

    The app object must implement get_step(idx), get_n_samples(idx), get_wave()
    so this engine stays decoupled from any specific sonification's data model.
    """

    def __init__(self, app):
        """Store app reference and initialise audio state."""
        self.app = app
        self.phase_by_track: dict[
            str, float
        ] = {}  # continuous phase per track to avoid clicks
        self._step_queue: queue.SimpleQueue[Step] = queue.SimpleQueue()
        self._remaining = 0  # samples left in the current step
        self._current: Step | None = None
        self.stream = None

    # ------------------------------------------------------------------ UI thread

    def start(self):
        """Pre-fill the queue and open the sounddevice output stream."""
        if self.stream:
            return
        for _ in range(PREFILL):
            self._enqueue_next()
        self.stream = sd.OutputStream(  # IMPORTANT: THE CONTINUOUS LOOP HAPPENS HERE: WE ARE TELLING sounddevice THAT IF IT NEEDS AUDIO IT SHOULD CALL _callback
            samplerate=FS,
            channels=1,
            dtype="float32",
            blocksize=STREAM_BLOCKSIZE,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        """Stop and close the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _enqueue_next(self):
        """
        Compute one Step from the current app state and push it onto the queue.

        Delegates data→frequency mapping to app.get_step(idx) and timing to
        app.get_n_samples(idx), so this method is sonification-agnostic.
        Advances app.index and schedules a plot redraw on the main thread.
        Called from the UI thread only (directly or via root.after).
        """
        app = self.app
        idx = app.index
        tracks = app.get_step(idx)  # (name, freq, amp) per active track
        n_samples = app.get_n_samples(idx)
        wave = app.get_wave()
        self._step_queue.put(Step(n_samples, tracks, wave, idx))
        app.index = (app.index + 1) % app.length
        app.root.after(0, app.update_plot)

    # ---------------------------------------------------------------- Audio thread

    @staticmethod  # ideal for utility or helper functions that perform tasks independent (notice there is no self here :) ) of object state
    def _oscillator(wave: str, phase: np.ndarray) -> np.ndarray:
        """
        Evaluate one cycle of the chosen waveform at the given phase values.

        All waveforms are closed-form numpy expressions — no loops,
        branchless per sample.  Returns values in [-1, 1].
        """
        t = (phase / (2 * np.pi)) % 1  # normalised phase in [0, 1)
        match wave:
            case "sinusoida":
                return np.sin(phase)
            case "trójkąt":
                return 2 * np.abs(2 * t - 1) - 1
            case "piła":
                return 2 * t - 1
            case "prostokąt":
                return np.where(np.sin(phase) >= 0, 1.0, -1.0)
            case _:
                return np.sin(phase)

    @staticmethod
    def render_tracks(
        tracks,
        wave,
        n,
        phase_by_track,
    ):
        """
        Render one audio block from oscillator tracks.
        """

        t = np.arange(n, dtype=np.float32) / FS

        mixed = np.zeros(n, np.float32)

        for name, freq, amp in tracks:
            phase0 = phase_by_track.get(name, 0.0)

            phase = phase0 + 2 * np.pi * freq * t

            osc = ContinuousSynth._oscillator(
                wave,
                phase,
            ).astype(np.float32)

            mixed += amp * osc

            phase_by_track[name] = (phase0 + 2 * np.pi * freq * n / FS) % (2 * np.pi)

        return soft_limiter(
            mixed,
            SOFT_LIMITER_DRIVE,
            SOFT_LIMITER_CEILING,
        )

    def _render(self, frames: int) -> np.ndarray:
        """
        Fill an output buffer of `frames` float32 samples.

        Consumes Steps from the queue one slice at a time; a single callback
        call may span multiple Steps at high BPM.  Silence on underflow.
        Phase is tracked per track so waveforms are click-free at boundaries.
        Called exclusively from the sounddevice audio thread.
        """
        out, pos = np.zeros(frames, np.float32), 0
        while pos < frames:
            if self._remaining <= 0:
                try:
                    self._current = self._step_queue.get_nowait()
                    self._remaining = self._current.samples
                    self.app.playing_index = self._current.index
                    self.app.root.after(0, self._enqueue_next)
                    note_str = (
                        "\n".join(
                            f"{n}: {freq:.1f} Hz | amp={amp:.2f}"
                            for n, freq, amp in self._current.tracks
                        )
                        or "—"
                    )
                    self.app.root.after(
                        0, lambda s=note_str: self.app.current_note_var.set(s)
                    )
                except queue.Empty:
                    break
            n = min(frames - pos, self._remaining)
            step = self._current
            if step.tracks:
                chunk = self.render_tracks(
                    tracks=step.tracks,
                    wave=step.wave,
                    n=n,
                    phase_by_track=self.phase_by_track,
                )

                out[pos : pos + n] = chunk
            self._remaining -= n
            pos += n
        return out

    def _flush_and_refill(self):
        """
        Drain the queue and refill it from the currently playing position.

        Called on any UI parameter change.  Restarts from playing_index + 1
        so changes take effect on the very next step with no rewind artefact.
        No-op when the stream is stopped.
        """
        while not self._step_queue.empty():
            try:
                self._step_queue.get_nowait()
            except queue.Empty:
                break
        if not self.stream:
            return
        self.app.index = (self.app.playing_index + 1) % self.app.length
        for _ in range(PREFILL):
            self._enqueue_next()

    def _callback(self, outdata, frames, time, status):
        """
        sounddevice audio callback — fires on the audio thread every blocksize frames.

        Must never block.  Delegates all work to _render().
        """
        if status:
            print(status)
        outdata[:, 0] = self._render(frames)
