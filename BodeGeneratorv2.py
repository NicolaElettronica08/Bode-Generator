# NICOLA MASSINI - Bode Generator - v 2.0

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from scipy import signal
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
from collections import Counter


class BodeApp:
    FIXED_POINTS = 100

    def bode_asymptotic(self, num, den, w):
        zeros = np.roots(num)
        poles = np.roots(den)
        mag = np.zeros_like(w, dtype=float)
        phase = np.zeros_like(w, dtype=float)
        k = num[0] / den[0]
        mag += 20 * np.log10(abs(k))
        if k < 0:
            phase += 180

        for z in zeros:
            if np.isclose(z, 0):
                mag += 20 * np.log10(w)
                phase += 90
        for p in poles:
            if np.isclose(p, 0):
                mag -= 20 * np.log10(w)
                phase -= 90

        for z in zeros:
            wz = abs(z)
            if wz == 0:
                continue
            mag += np.where(w >= wz, 20 * np.log10(w / wz), 0)
            phase += np.piecewise(
                w,
                [w < wz / 10, (w >= wz / 10) & (w <= 10 * wz), w > 10 * wz],
                [0, lambda x: 45 * (np.log10(x / wz) + 1), 90],
            )

        for p in poles:
            wp = abs(p)
            if wp == 0:
                continue
            mag -= np.where(w >= wp, 20 * np.log10(w / wp), 0)
            phase += np.piecewise(
                w,
                [w < wp / 10, (w >= wp / 10) & (w <= 10 * wp), w > 10 * wp],
                [0, lambda x: -45 * (np.log10(x / wp) + 1), -90],
            )
        return mag, phase

    def __init__(self, root):
        self.root = root
        self.root.title("Bode Generator v2.0")
        self.root.geometry("1550x800")
        self.root.iconbitmap("bode.ico")
        self.root.resizable(True, True)
        self.root.bind("<Return>", lambda event: self.plot_bode())

        self.current_w = None
        self.current_mag = None
        self.current_phase = None
        self.cursor_positions = [1.0, 10.0]
        self.dragging_cursor = None

        control_frame = ttk.Frame(root, padding=10, relief="groove", borderwidth=2)
        control_frame.pack(side="left", fill="y")

        info_frame = ttk.Frame(root, padding=10, relief="groove", borderwidth=2)
        info_frame.pack(side="right", fill="y")

        plot_frame = ttk.Frame(root, padding=10)
        plot_frame.pack(side="left", fill="both", expand=True)

        ttk.Label(control_frame, text="Funzione di trasferimento F(s):").pack(anchor="w")
        self.tf_entry = ttk.Entry(control_frame)
        self.tf_entry.pack(fill="x", pady=5)

        ttk.Label(control_frame, text="Tipo diagramma:").pack(anchor="w")
        self.diagram_type = ttk.Combobox(control_frame, values=["Reale", "Asintotico"], state="readonly")
        self.diagram_type.set("Reale")
        self.diagram_type.pack(fill="x", pady=5)

        self.show_both = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Mostra reale + asintotico", variable=self.show_both).pack(anchor="w", pady=5)

        ttk.Label(control_frame, text="Esponente pulsazione iniziale log10(ω):").pack(anchor="w")
        self.sfe_entry = ttk.Entry(control_frame)
        self.sfe_entry.insert(0, "0")
        self.sfe_entry.pack(fill="x", pady=5)

        ttk.Label(control_frame, text="Esponente pulsazione finale log10(ω):").pack(anchor="w")
        self.efe_entry = ttk.Entry(control_frame)
        self.efe_entry.insert(0, "9")
        self.efe_entry.pack(fill="x", pady=5)

        ttk.Label(control_frame, text="Scala verticale modulo [min,max]:").pack(anchor="w")
        self.mag_min_entry = ttk.Entry(control_frame)
        self.mag_min_entry.pack(fill="x", pady=2)
        self.mag_max_entry = ttk.Entry(control_frame)
        self.mag_max_entry.pack(fill="x", pady=2)

        ttk.Label(control_frame, text="Scala verticale fase [min,max]:").pack(anchor="w")
        self.phase_min_entry = ttk.Entry(control_frame)
        self.phase_min_entry.pack(fill="x", pady=2)
        self.phase_max_entry = ttk.Entry(control_frame)
        self.phase_max_entry.pack(fill="x", pady=2)

        ttk.Button(control_frame, text="Genera diagramma", command=self.plot_bode).pack(fill="x", pady=5)
        ttk.Button(control_frame, text="Esporta grafico", command=self.export_graph).pack(fill="x", pady=5)

        ttk.Label(info_frame, text="Poli e Zeri").pack()
        self.pz_text = tk.Text(info_frame, width=34, height=18)
        self.pz_text.pack(fill="x", pady=5)

        ttk.Label(info_frame, text="Valori cursori").pack()
        self.cursor_text = tk.Text(info_frame, width=34, height=18)
        self.cursor_text.pack(fill="x", pady=5)

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def update_cursor_info(self):
        if self.current_w is None:
            return

        self.cursor_text.delete("1.0", tk.END)
        values = []

        for i, x in enumerate(self.cursor_positions, start=1):
            idx = np.argmin(np.abs(self.current_w - x))
            w = self.current_w[idx]
            f = w / (2 * np.pi)
            mag = self.current_mag[idx]
            ph = self.current_phase[idx]
            values.append((w, f, mag, ph))

            self.cursor_text.insert(
                tk.END,
                f"Cursore {i}\nω = {w:.5g} rad/s\nf = {f:.5g} Hz\n|G| = {mag:.3f} dB\n∠G = {ph:.3f}°\n\n",
            )

        if len(values) == 2:
            dw = abs(values[1][0] - values[0][0])
            df = abs(values[1][1] - values[0][1])
            dmag = values[1][2] - values[0][2]
            dph = values[1][3] - values[0][3]
            self.cursor_text.insert(
                tk.END,
                f"Differenze\nΔω = {dw:.5g} rad/s\nΔf = {df:.5g} Hz\nΔ|G| = {dmag:.3f} dB\nΔ∠G = {dph:.3f}°\n",
            )

    def on_press(self, event):
        if event.inaxes not in [self.ax1, self.ax2] or event.xdata is None:
            return
        distances = [abs(np.log10(max(x, 1e-30)) - np.log10(max(event.xdata, 1e-30))) for x in self.cursor_positions]
        self.dragging_cursor = int(np.argmin(distances))

    def on_motion(self, event):
        if self.dragging_cursor is None or event.inaxes not in [self.ax1, self.ax2] or event.xdata is None:
            return
        self.cursor_positions[self.dragging_cursor] = max(event.xdata, 1e-12)
        self.draw_cursors()
        self.update_cursor_info()

    def on_release(self, event):
        self.dragging_cursor = None

    def draw_cursors(self):
        for ax in [self.ax1, self.ax2]:
            removable = [ln for ln in ax.lines if getattr(ln, "_is_cursor", False)]
            for ln in removable:
                ln.remove()

        for idx, x in enumerate(self.cursor_positions):
            color = "red" if idx == 0 else "green"
            for ax in [self.ax1, self.ax2]:
                line = ax.axvline(x=x, linewidth=1.8, color=color)
                line._is_cursor = True

        self.canvas.draw_idle()

    def plot_bode(self):
        try:
            expr = self.tf_entry.get().strip()
            s = sp.symbols("s")
            num_expr, den_expr = sp.fraction(sp.simplify(sp.sympify(expr)))
            num = [float(c) for c in sp.Poly(num_expr, s).all_coeffs()]
            den = [float(c) for c in sp.Poly(den_expr, s).all_coeffs()]

            tf = signal.TransferFunction(num, den)
            min_w = 10 ** int(self.sfe_entry.get())
            max_w = 10 ** int(self.efe_entry.get())
            w = np.logspace(np.log10(min_w), np.log10(max_w), num=self.FIXED_POINTS)

            _, mag_real, phase_real = signal.bode(tf, w)
            mag_asym, phase_asym = self.bode_asymptotic(num, den, w)

            mag_asym += mag_real[0] - mag_asym[0]
            phase_asym += phase_real[0] - phase_asym[0]

            self.ax1.clear()
            self.ax2.clear()

            if self.show_both.get():
                self.ax1.semilogx(w, mag_real, linewidth=2, label="Reale")
                self.ax1.semilogx(w, mag_asym, linewidth=2, label="Asintotico")
                self.ax2.semilogx(w, phase_real, linewidth=2, label="Reale")
                self.ax2.semilogx(w, phase_asym, linewidth=2, label="Asintotico")
                self.ax1.legend()
                self.ax2.legend()
                self.current_mag = mag_real
                self.current_phase = phase_real
            else:
                if self.diagram_type.get() == "Reale":
                    self.ax1.semilogx(w, mag_real, linewidth=2)
                    self.ax2.semilogx(w, phase_real, linewidth=2)
                    self.current_mag = mag_real
                    self.current_phase = phase_real
                else:
                    self.ax1.semilogx(w, mag_asym, linewidth=2)
                    self.ax2.semilogx(w, phase_asym, linewidth=2)
                    self.current_mag = mag_asym
                    self.current_phase = phase_asym

            self.current_w = w

            zeros = np.roots(num)
            poles = np.roots(den)
            self.pz_text.delete("1.0", tk.END)
            z_count = Counter([complex(round(z.real, 5), round(z.imag, 5)) for z in zeros])
            p_count = Counter([complex(round(p.real, 5), round(p.imag, 5)) for p in poles])

            self.pz_text.insert(tk.END, "Zeri:\n")
            for val, cnt in z_count.items():
                self.pz_text.insert(tk.END, f"{val} (molteplicità {cnt})\n")

            self.pz_text.insert(tk.END, "\nPoli:\n")
            for val, cnt in p_count.items():
                self.pz_text.insert(tk.END, f"{val} (molteplicità {cnt})\n")

            try:
                self.ax1.set_ylim(float(self.mag_min_entry.get()), float(self.mag_max_entry.get()))
            except:
                pass

            try:
                self.ax2.set_ylim(float(self.phase_min_entry.get()), float(self.phase_max_entry.get()))
            except:
                pass

            self.ax1.set_xlim(min_w, max_w)
            self.ax2.set_xlim(min_w, max_w)
            self.ax1.margins(x=0)
            self.ax2.margins(x=0)
            self.ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
            self.ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
            self.ax1.minorticks_on()
            self.ax2.minorticks_on()
            self.ax2.set_xlabel("Pulsazione ω [rad/s]")
            self.ax1.set_ylabel("Modulo [dB]")
            self.ax2.set_ylabel("Fase [°]")

            self.cursor_positions = [min_w * 1.2, min_w * 2.0]
            self.draw_cursors()
            self.update_cursor_info()

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Errore", str(e))

    def export_graph(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf")],
        )

        if file_path:
            saved_positions = list(self.cursor_positions)
            self.cursor_positions = []
            self.draw_cursors()

            self.fig.savefig(
                file_path,
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.05,
            )

            self.cursor_positions = saved_positions
            self.draw_cursors()


if __name__ == "__main__":
    root = tk.Tk()
    app = BodeApp(root)
    root.mainloop()

