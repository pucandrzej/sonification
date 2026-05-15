# CZY SŁYCHAĆ CZASAMI "KLIKANIE"? WYNIKA ONO NAJPEWNIEJ Z NIECIĄGŁOŚCI - MOŻNA DODAĆ FADE IN / FADE OUT ŻEBY GO UNIKNĄĆ

"""
sonifikacja_obrazu.py — real-time image sonification
=====================================================

IDEA
----
A greyscale image is treated as a spectrogram-like grid:
- X axis = time (columns = sixteenth-note steps)
- Y axis = pitch (rows mapped to scale degrees, bottom = low, top = high)
- Pixel brightness = amplitude of that note at that step

At each step, every row that maps to an active scale degree is played as a
sine (or chosen waveform) at the corresponding frequency, with amplitude
proportional to the pixel brightness in that row.

ARCHITECTURE
------------
Reuses Step and ContinuousSynth from synth_engine.py unchanged.
This app implements the three delegate methods the engine calls:
    get_step(idx)      → build (name, freq, amp) from the image column
    get_n_samples(idx) → timing from BPM
    get_wave()         → waveform from UI

Image pipeline:
    PIL open → greyscale → resize to (n_steps × n_pitches) → numpy float32 [0,1]
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from synth_engine import ContinuousSynth
from utils import (
    FS,
    SWING_CLIP,
    NOTE_OFFSETS,
    SCALES,
    AMP_TOTAL,
    DEFAULT_NOTE,
    DEFAULT_GLOBAL_OCTAVE,
    DEFAULT_SCALE,
    DEFAULT_WAVE,
    DEFAULT_BPM,
    BPM_RANGE,
    sixteenth_duration,
    next_numbered_wav,
    save_wav,
    semitone_to_frequency,
    THUMBNAIL_SIZE,
    IMAGE_QUANTIZATION_LEVELS,
    STEPS_OPTIONS,
    OCTAVE_RANGE_OPTIONS,
    SEMITONES_NO,
    GREYSCALE_NORMALIZATION_CONST,
)


class SonifikacjaObrazu:
    """
    Image sonification app.

    The image is resized to (n_steps × n_pitches) where n_pitches is the
    number of scale degrees in the chosen octave range.  Each column is one
    Step; each row is one pitch; brightness drives amplitude.
    """

    def __init__(self, root):
        """Initialise state, build UI, wire traces."""
        self.root = root
        self.root.title("Sonifikacja obrazu")

        # audio position
        self.index = 0
        self.playing_index = 0

        # image state — set by _load_image
        self.image_pixels: np.ndarray | None = (
            None  # shape (n_pitches, n_steps) float32 [0,1]
        )
        self.length = None  # columns, updated when image or steps_var changes

        # UI variables
        self.note_var = tk.StringVar(value=DEFAULT_NOTE)
        self.global_octave_var = tk.IntVar(value=DEFAULT_GLOBAL_OCTAVE)
        self.scale_var = tk.StringVar(value=DEFAULT_SCALE)
        self.wave_var = tk.StringVar(value=DEFAULT_WAVE)
        self.bpm_var = tk.DoubleVar(value=DEFAULT_BPM)
        self.steps_var = tk.IntVar(value=STEPS_OPTIONS[0])
        self.octave_range_var = tk.IntVar(value=OCTAVE_RANGE_OPTIONS[0])
        self.current_note_var = tk.StringVar(value="—")

        self.synth = ContinuousSynth(self)

        for var in [
            self.note_var,
            self.global_octave_var,
            self.scale_var,
            self.wave_var,
            self.bpm_var,
            self.steps_var,
            self.octave_range_var,
        ]:
            var.trace_add("write", lambda *_: self._on_param_change())

        self._build_ui()
        self.original_image = None
        self._ui_refresh()

    def _ui_refresh(self):
        if hasattr(self, "_plot_bg"):  # guard: image may not be loaded yet
            self._update_playhead()
        self.root.after(5, self._ui_refresh)

    # ---------------------------------------------------------------- engine protocol

    def get_step(self, idx) -> tuple:
        """
        Convert one image column into active oscillator tracks.
        """
        if self.image_pixels is None:
            return ()

        col = idx % self.image_pixels.shape[1]  # boundary condition handled via %

        pitches = self._pitch_freqs()

        column = self.image_pixels[:, col]

        tracks = []

        active_pixels = np.count_nonzero(column)

        if active_pixels == 0:
            return ()

        total_amp = AMP_TOTAL / len(column)

        for row_idx, (freq, brightness) in enumerate(zip(pitches, column)):
            tracks.append(
                (
                    f"row{row_idx}",
                    freq,
                    total_amp * brightness,
                )
            )

        return tuple(tracks)

    def get_n_samples(self, idx) -> int:
        """Return sample count for one sixteenth-note step at current BPM."""
        return max(
            1, int(sixteenth_duration(self.bpm_var.get(), SWING_CLIP[0], idx) * FS)
        )

    def get_wave(self) -> str:
        """Return the current waveform name."""
        return self.wave_var.get()

    # ---------------------------------------------------------------- image helpers

    def _pitch_freqs(self) -> list[float]:
        """
        Return one frequency per image row.

        Row 0 = LOWEST pitch.
        Last row = HIGHEST pitch.
        """
        scale = SCALES[self.scale_var.get()]
        octave_range = self.octave_range_var.get()

        if scale is None:
            semitones = list(range(octave_range * SEMITONES_NO + 1))
        else:
            semitones = []
            for oct_shift in range(octave_range):
                semitones += [s + SEMITONES_NO * oct_shift for s in scale]

        semitones = sorted(set(semitones))

        return [
            semitone_to_frequency(
                semitone=s,
                note=self.note_var.get(),
                octave_shift=int(self.global_octave_var.get()),
            )
            for s in semitones
        ]

    def _load_image(self, path: str):
        """
        Load image and convert it into a quantized spectrogram grid.
        """
        if not path:
            return

        # store ORIGINAL image permanently
        self.original_image = Image.open(path).convert("L")

        n_steps = self.steps_var.get()
        n_pitches = len(self._pitch_freqs())

        # flip once so:
        # bottom = low pitch
        # top = high pitch
        img = self.original_image.transpose(Image.FLIP_TOP_BOTTOM)

        img = img.resize((n_steps, n_pitches), Image.LANCZOS)

        arr = np.array(img, dtype=np.float32) / GREYSCALE_NORMALIZATION_CONST

        # ----------------------------
        # QUANTIZATION
        # ----------------------------

        levels = IMAGE_QUANTIZATION_LEVELS

        arr = np.round(arr * (levels - 1)) / (levels - 1)

        self.image_pixels = arr

        self.length = n_steps
        self.index = 0
        self.playing_index = 0

        self._refresh_image_display()
        self.update_plot()

    def _refresh_image_display(self):
        """
        Display the ORIGINAL image preview at the top of the UI.

        This is purely visual and does NOT affect the sonification grid.
        """
        if self.original_image is None:
            return

        preview = self.original_image.copy()

        preview.thumbnail(THUMBNAIL_SIZE)

        self._tk_image = ImageTk.PhotoImage(preview)

        self._image_label.configure(image=self._tk_image)

    def _on_param_change(self):
        """
        Rebuild quantized spectrogram from ORIGINAL image.
        """
        if self.original_image is None:
            return

        n_steps = self.steps_var.get()
        n_pitches = len(self._pitch_freqs())

        img = self.original_image.transpose(Image.FLIP_TOP_BOTTOM)

        img = img.resize((n_steps, n_pitches), Image.LANCZOS)  # resampling Lanczosa

        arr = np.array(img, dtype=np.float32) / GREYSCALE_NORMALIZATION_CONST

        levels = IMAGE_QUANTIZATION_LEVELS

        arr = np.round(arr * (levels - 1)) / (levels - 1)

        self.image_pixels = arr

        self.length = n_steps

        self._refresh_image_display()

        self.synth._flush_and_refill()

        self.update_plot()

    # ---------------------------------------------------------------- UI

    def _build_ui(self):
        """
        Build the layout.

        Left: scrollable controls.  Right: image preview + playhead plot.
        """
        outer = ttk.Frame(self.root)
        outer.pack(side=tk.LEFT, fill=tk.Y)

        cv = tk.Canvas(outer, width=200, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cv.pack(side=tk.LEFT, fill=tk.Y, expand=True)

        c = ttk.Frame(cv, padding=8)
        wid = cv.create_window((0, 0), window=c, anchor="nw")
        c.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(wid, width=e.width))
        cv.bind_all(
            "<MouseWheel>", lambda e: cv.yview_scroll(int(-e.delta / 120), "units")
        )
        cv.bind_all("<Button-4>", lambda e: cv.yview_scroll(-1, "units"))
        cv.bind_all("<Button-5>", lambda e: cv.yview_scroll(1, "units"))

        ttk.Button(
            c,
            text="WCZYTAJ OBRAZ",
            command=lambda: self._load_image(
                filedialog.askopenfilename(
                    filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
                )
            ),
        ).pack(fill=tk.X, pady=2)

        for text, cmd in [
            ("START", self.synth.start),
            ("STOP", self.synth.stop),
            ("ZAPISZ PĘTLĘ WAV", self.export_loop),
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

        ttk.Label(c, text="Tempo BPM").pack(anchor="w")
        ttk.Scale(
            c,
            from_=BPM_RANGE[0],
            to=BPM_RANGE[1],
            variable=self.bpm_var,
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X)
        ttk.Label(c, textvariable=self.bpm_var).pack(anchor="w")

        ttk.Separator(c).pack(fill=tk.X, pady=8)

        for label, var, values in [
            ("Ton bazowy", self.note_var, list(NOTE_OFFSETS)),
            ("Oktawa globalna od C3", self.global_octave_var, [-2, -1, 0, 1, 2]),
            ("Tryb skali", self.scale_var, list(SCALES)),
            (
                "Przebieg fali",
                self.wave_var,
                ["sinusoida", "trójkąt", "piła", "prostokąt"],
            ),
            ("Liczba kroków", self.steps_var, STEPS_OPTIONS),
            ("Zakres oktaw", self.octave_range_var, OCTAVE_RANGE_OPTIONS),
        ]:
            ttk.Label(c, text=label).pack(anchor="w")
            ttk.Combobox(c, textvariable=var, values=values, state="readonly").pack(
                fill=tk.X
            )

        # right panel: image preview + plot
        right = ttk.Frame(self.root, padding=4)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._image_label = ttk.Label(right, text="(brak obrazu)")
        self._image_label.pack(pady=4)

        self.fig = Figure(figsize=(9, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("resize_event", lambda e: self._build_plot())

    def _build_plot(self):
        """Draw static content (image heatmap + grid lines) and save background for blitting."""
        self.ax.clear()
        if self.image_pixels is not None:
            self.ax.imshow(
                self.image_pixels,
                aspect="auto",
                origin="lower",
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            for y in range(self.image_pixels.shape[0]):
                self.ax.axhline(y - 0.5, linewidth=0.3)
        self.ax.set(
            title="Obraz jako spektrogram",
            xlabel="krok (czas)",
            ylabel="rząd (wysokość dźwięku)",
        )
        self.canvas.draw()
        self._plot_bg = self.canvas.copy_from_bbox(self.ax.bbox)
        self._vline = self.ax.axvline(
            self.playing_index, color="red", linestyle="--", animated=True
        )

    def update_plot(self):
        """Alias so external callers (_load_image, _on_param_change) trigger a full rebuild."""
        self._build_plot()

    def _update_playhead(self):
        """Blit only the vertical line — no full redraw."""
        self.canvas.restore_region(self._plot_bg)
        self._vline.set_xdata([self.playing_index, self.playing_index])
        self.ax.draw_artist(self._vline)
        self.canvas.blit(self.ax.bbox)

    def export_loop(self):
        """
        Render the full image loop offline and save as a numbered WAV file.

        Uses the shared render_tracks() helper so offline rendering
        matches realtime playback exactly.
        """
        if self.image_pixels is None:
            messagebox.showwarning(
                "Brak obrazu",
                "Najpierw wczytaj obraz.",
            )
            return

        phase_by_track = {}

        blocks = []

        for i in range(self.length):
            tracks = self.get_step(i)

            n = self.get_n_samples(i)

            block = ContinuousSynth.render_tracks(
                tracks=tracks,
                wave=self.get_wave(),
                n=n,
                phase_by_track=phase_by_track,
            )

            blocks.append(block)

        audio = np.concatenate(blocks)

        path = next_numbered_wav(prefix="image_loop")

        save_wav(path, audio, FS)

        messagebox.showinfo(
            "Zapisano",
            f"Zapisano plik:\n{path}",
        )


def main():
    """Initialise tkinter root, create the app, start the event loop."""
    root = tk.Tk()
    app = SonifikacjaObrazu(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.synth.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
