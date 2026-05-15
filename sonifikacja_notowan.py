"""
Sonifikacja notowań — real-time market data sonification
=========================================================

IDEA
----

UI thread
v
creates Step
v
puts Step into queue
v
audio thread reads Step

Normalised time-series data (e.g. crypto prices) is mapped to musical
pitches and played back in real time. Each data point becomes one
sixteenth-note step; the user can tweak all musical parameters live while
playback continues without interruption.

ARCHITECTURE — HOW THE PIECES FIT TOGETHER
-------------------------------------------

1. DATA LAYER
   JSON file - dict loaded once at startup.
   Each series is pre-normalised to [0, 1] (raw and derivative).
   `series_value()` selects which version to read at runtime.

2. UI LAYER  (main thread, tkinter)
   - All controls are tkinter Variables (StringVar, IntVar, DoubleVar,
     BooleanVar).  The UI never touches audio state directly.
   - `trace_add("write", ...)` on every variable fires callbacks the moment
     any control changes — no polling, no "Apply" button needed.
   - The plot is a matplotlib Figure embedded via FigureCanvasTkAgg.
     `canvas.draw_idle()` redraws it safely from the main thread.
   - The left panel is a scrollable region built with a plain tk.Canvas +
     Scrollbar wrapping a ttk.Frame (standard tkinter scroll pattern, since
     ttk has no native scrollable frame).

3. AUDIO LAYER  (sounddevice callback thread)
   sounddevice opens an OutputStream that fires `_callback` on a dedicated
   OS audio thread every ~few milliseconds, requesting a fixed number of
   samples (`blocksize`).  That callback must never block.

4. THREAD-SAFE HANDOFF — queue.SimpleQueue
   The classic producer/consumer pattern:
   - PRODUCER (main thread): `_enqueue_next()` pre-computes the next Step
     (frequencies, amplitudes, sample count) and puts it on a SimpleQueue.
   - CONSUMER (audio thread): `_render()` pops Steps with `get_nowait()`
     (non-blocking) and synthesises samples.
   The queue is pre-filled with 8 steps at start() to prevent underflows.

5. PARAMETER CHANGES WITHOUT GLITCHES — _flush_and_refill()
   When the user changes any control, `_flush_and_refill()` drains the
   queue and refills it from `playing_index + 1` (the step currently in
   the DAC, not the pre-computed future).  This makes changes audible on
   the very next step with no rewind.

6. SYNC BETWEEN AUDIO AND PLOT
   `playing_index` is written by the audio thread when it pops a Step, and
   read by the main thread to draw the vertical playhead line.

"""

import json
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from utils import (
    MARKET_DATA_FILE,
    SWING_CLIP,
    FS,
    NOTE_OFFSETS,
    SCALES,
    AMP_TOTAL,
    DEFAULT_NOTE,
    DEFAULT_GLOBAL_OCTAVE,
    DEFAULT_SCALE,
    DEFAULT_WAVE,
    DEFAULT_DATA_MODE,
    DEFAULT_BPM,
    WAV_EXPORT_PREFIX,
    PLOT_YLIM,
    TRACK_OCTAVE_VALUES,
    GLOBAL_OCTAVE_VALUES,
    BPM_RANGE,
    value_to_frequency,
    sixteenth_duration,
    next_numbered_wav,
    save_wav,
)

from synth_engine import ContinuousSynth


class SonifikacjaNotowan:
    """
    Main application class.

    Owns all tkinter Variables, the data dict, the synth, and the plot.
    Acts as the shared state store that both the UI and ContinuousSynth read from.
    """

    def __init__(self, root):
        """Load data, initialise all state variables, wire up traces, build UI."""
        self.root = root
        self.root.title("Sonifikacja notowań — limiter")
        with open(MARKET_DATA_FILE, encoding="utf-8") as f:
            self.data = json.load(f)["coins"]
        self.names = list(self.data)
        self.length = min(
            len(self.data[n]["normalized"]) for n in self.names
        )  # common length
        self.index = 0  # next step to enqueue
        self.playing_index = (
            0  # step currently audible (written by audio thread, read by UI)
        )

        self.note_var = tk.StringVar(value=DEFAULT_NOTE)
        self.global_octave_var = tk.IntVar(value=DEFAULT_GLOBAL_OCTAVE)
        self.scale_var = tk.StringVar(value=DEFAULT_SCALE)
        self.wave_var = tk.StringVar(value=DEFAULT_WAVE)
        self.mode_var = tk.StringVar(value=DEFAULT_DATA_MODE)
        self.bpm_var = tk.DoubleVar(value=DEFAULT_BPM)
        self.swing_var = tk.DoubleVar(value=SWING_CLIP[0])
        self.track_enabled = {n: tk.BooleanVar(value=True) for n in self.names}
        self.track_octave = {n: tk.IntVar(value=0) for n in self.names}
        self.current_note_var = tk.StringVar(value="—")

        self.synth = ContinuousSynth(self)

        # trace_add fires _flush_and_refill on every control change so audio
        # updates immediately without restarting the stream
        for var in [
            self.note_var,
            self.global_octave_var,
            self.scale_var,
            self.wave_var,
            self.mode_var,
            self.bpm_var,
            self.swing_var,
            *self.track_enabled.values(),
            *self.track_octave.values(),
        ]:
            var.trace_add("write", lambda *_: self.synth._flush_and_refill())

        # these vars also need to refresh the plot even when stopped
        for var in [self.mode_var, *self.track_enabled.values()]:
            var.trace_add("write", lambda *_: self.update_plot())

        self._build_ui()
        self.update_plot()

    def get_step(self, idx) -> tuple:
        """Return (name, freq, amp) for each active track at data index idx."""
        active = [n for n in self.names if self.track_enabled[n].get()]
        amp = AMP_TOTAL / max(1, len(active))
        return tuple(
            (
                n,
                value_to_frequency(
                    value=self.series_value(n, idx),
                    note=self.note_var.get(),
                    global_octave=int(self.global_octave_var.get()),
                    track_octave=int(self.track_octave[n].get()),
                    scale_name=self.scale_var.get(),
                ),
                amp,
            )
            for n in active
        )

    def get_n_samples(self, idx) -> int:
        """Return the number of audio samples for step idx at current BPM/swing."""
        return max(
            1,
            int(sixteenth_duration(self.bpm_var.get(), self.swing_var.get(), idx) * FS),
        )

    def get_wave(self) -> str:
        """Return the current waveform name."""
        return self.wave_var.get()

    def series_value(self, name, index):
        """Return the normalised data value for a track at a given step index."""
        key = "normalized" if self.mode_var.get() == "kurs" else "normalized_derivative"
        values = np.asarray(self.data[name][key], float)
        return float(
            values[index % len(values)]
        )  # IMPORTANT: % is used here to wrap the index

    def _build_ui(self):
        """
        Build the full UI layout.

        Left panel: scrollable controls inside a tk.Canvas + Scrollbar.
        The inner ttk.Frame is embedded via canvas.create_window() and resizes
        with the canvas via a <Configure> binding.  Mouse wheel scrolling is
        bound globally for Windows/macOS (<MouseWheel>) and Linux (<Button-4/5>).

        Right panel: matplotlib figure embedded via FigureCanvasTkAgg.
        """
        # scrollable left panel
        outer = ttk.Frame(self.root)
        outer.pack(side=tk.LEFT, fill=tk.Y)

        canvas = tk.Canvas(outer, width=200, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.Y, expand=True)

        c = ttk.Frame(canvas, padding=8)
        window_id = canvas.create_window((0, 0), window=c, anchor="nw")

        c.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width)
        )
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1 * e.delta / 120), "units"),
        )
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        for text, cmd in [
            ("START", self.synth.start),
            ("STOP", self.synth.stop),
            ("ZAPISZ JEDNĄ PĘTLĘ WAV", self.export_loop),
        ]:
            ttk.Button(c, text=text, command=cmd).pack(fill=tk.X, pady=2)

        ttk.Separator(c).pack(fill=tk.X, pady=8)

        ttk.Label(c, text="Aktualnie grane:").pack(anchor="w")
        ttk.Label(
            c,
            textvariable=self.current_note_var,
            font=("Courier", 9),
            justify=tk.LEFT,
            foreground="#1a6e2e",
        ).pack(anchor="w", pady=(0, 4))

        ttk.Separator(c).pack(fill=tk.X, pady=8)

        for label, var, lo, hi in [
            ("Tempo BPM", self.bpm_var, *BPM_RANGE),
            ("Swing %", self.swing_var, *SWING_CLIP),
        ]:
            ttk.Label(c, text=label).pack(anchor="w")
            ttk.Scale(c, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL).pack(
                fill=tk.X
            )
            ttk.Label(c, textvariable=var).pack(anchor="w")

        ttk.Separator(c).pack(fill=tk.X, pady=8)

        for label, var, values in [
            ("Ton bazowy", self.note_var, list(NOTE_OFFSETS)),
            ("Oktawa globalna od C3", self.global_octave_var, GLOBAL_OCTAVE_VALUES),
            ("Tryb skali", self.scale_var, list(SCALES)),
            ("Tryb danych", self.mode_var, ["kurs", "pochodna kursu"]),
            (
                "Przebieg fali",
                self.wave_var,
                ["sinusoida", "trójkąt", "piła", "prostokąt"],
            ),
        ]:
            ttk.Label(c, text=label).pack(anchor="w")
            ttk.Combobox(c, textvariable=var, values=values, state="readonly").pack(
                fill=tk.X
            )

        ttk.Separator(c).pack(fill=tk.X, pady=8)

        for name in self.names:
            f = ttk.LabelFrame(c, text=name)
            f.pack(fill=tk.X, pady=4)
            ttk.Checkbutton(
                f,
                text="włącz ścieżkę",
                variable=self.track_enabled[name],
                command=self.update_plot,
            ).pack(anchor="w")
            ttk.Label(f, text="oktawa ścieżki").pack(anchor="w")
            ttk.Combobox(
                f,
                textvariable=self.track_octave[name],
                values=TRACK_OCTAVE_VALUES,
                state="readonly",
                width=5,
            ).pack(anchor="w")

        plot_frame = ttk.Frame(self.root, padding=4)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(9, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self):
        """
        Redraw the matplotlib axes with current data and player position.

        Uses playing_index (not the enqueue-ahead index) so the vertical line
        matches what is actually audible.  draw_idle() defers the render to
        the next idle cycle, avoiding redundant redraws if called rapidly.
        """
        self.ax.clear()
        active = [n for n in self.names if self.track_enabled[n].get()]
        for name in active:
            key = (
                "normalized"
                if self.mode_var.get() == "kurs"
                else "normalized_derivative"
            )
            self.ax.plot(np.asarray(self.data[name][key], float), label=name)
        self.ax.axvline(self.playing_index, linestyle="--")
        self.ax.set(
            title="Aktywne przebiegi danych i aktualna pozycja",
            xlabel="krok",
            ylabel="wartość znormalizowana",
            ylim=PLOT_YLIM,
        )
        if active:
            self.ax.legend(loc="upper right")
        self.canvas.draw_idle()

    def export_loop(self):
        """
        Render the full data loop offline and save it as a numbered WAV file.

        Uses shared render_tracks() helper so export matches realtime playback.
        """

        phase_by_track = {}

        blocks = []

        for i in range(self.length):
            active = [n for n in self.names if self.track_enabled[n].get()]

            amp = AMP_TOTAL / max(1, len(active))

            tracks = []

            for name in active:
                freq = value_to_frequency(
                    value=self.series_value(name, i),
                    note=self.note_var.get(),
                    global_octave=int(self.global_octave_var.get()),
                    track_octave=int(self.track_octave[name].get()),
                    scale_name=self.scale_var.get(),
                )

                tracks.append(
                    (
                        name,
                        freq,
                        amp,
                    )
                )

            n = max(
                1,
                int(
                    sixteenth_duration(
                        self.bpm_var.get(),
                        self.swing_var.get(),
                        i,
                    )
                    * FS
                ),
            )

            block = ContinuousSynth.render_tracks(
                tracks=tracks,
                wave=self.wave_var.get(),
                n=n,
                phase_by_track=phase_by_track,
            )

            blocks.append(block)

        audio = np.concatenate(blocks)

        path = next_numbered_wav(prefix=WAV_EXPORT_PREFIX)

        save_wav(path, audio, FS)

        messagebox.showinfo(
            "Zapisano",
            f"Zapisano plik:\n{path}",
        )


def main():
    """Initialise the tkinter root, create the app, and start the event loop."""
    root = tk.Tk()
    app = SonifikacjaNotowan(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.synth.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
