##NICOLA MASSINI - Bode Generator - v 1.0

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from scipy import signal
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
from collections import Counter

class BodeApp:
    def toggle_points(self):
        if self.diagram_type.get() == "Asintotico":
            self.n_entry.config(state="disabled")
        else:
            self.n_entry.config(state="normal")
    
    def auto_frequency_range(self, num, den):
        zeros = np.roots(num)
        poles = np.roots(den)
        freqs = [abs(x) for x in np.concatenate((zeros, poles)) if abs(x) > 0]
        if not freqs:
            return 0.01, 100
        min_w = min(freqs)/10
        max_w = max(freqs)*10
        return min_w, max_w
    
    def bode_asymptotic(self, num, den, w):
        zeros = np.roots(num)
        poles = np.roots(den)
        mag = np.zeros_like(w, dtype=float)
        phase = np.zeros_like(w, dtype=float)
        k = num[0]/den[0]
        mag += 20*np.log10(abs(k))
        if k < 0:
            phase += 180

        for z in zeros:
            if np.isclose(z,0):
                mag += 20*np.log10(w)
                phase += 90
        for p in poles:
            if np.isclose(p,0):
                mag -= 20*np.log10(w)
                phase -= 90

        for z in zeros:
            wz = abs(z)
            if wz == 0:
                continue
            mag += np.where(w>=wz, 20*np.log10(w/wz), 0)
            phase += np.piecewise(
                w,
                [w < wz/10, (w >= wz/10) & (w <= 10*wz), w > 10*wz],
                [0, lambda x: 45*(np.log10(x/wz)+1), 90],
            )

        for p in poles:
            wp = abs(p)
            if wp == 0:
                continue
            mag -= np.where(w>=wp, 20*np.log10(w/wp),0)
            phase += np.piecewise(
                w,
                [w < wp/10, (w >= wp/10) & (w <= 10*wp), w > 10*wp],
                [0, lambda x: -45*(np.log10(x/wp)+1), -90],
            )
        return mag, phase
    
    def __init__(self, root):
        self.root = root
        self.root.title("Bode Generator")
        self.root.geometry("1400x750")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background='#f0f0f0', foreground='#000', font=('Arial',10))
        style.configure('TButton', padding=5)
        style.configure('TEntry', padding=5)
        style.configure('TLabel', padding=5)
        style.configure('TCheckbutton', padding=5)
        style.configure('TCombobox', padding=5)

        # Frame controllo
        control_frame = ttk.Frame(root, padding=10, relief='groove', borderwidth=2)
        control_frame.pack(side="left", fill="y")

        # Frame per lista poli/zeri
        polezero_frame = ttk.Frame(root, padding=10, relief='groove', borderwidth=2)
        polezero_frame.pack(side="right", fill="y")

        plot_frame = ttk.Frame(root, padding=10)
        plot_frame.pack(side="left", fill="both", expand=True)

        ttk.Label(control_frame, text="Funzione di trasferimento F(s):").pack(anchor="w")
        self.tf_entry = ttk.Entry(control_frame)
        self.tf_entry.insert(0,"")
        self.tf_entry.pack(fill="x", pady=5)

        ttk.Label(control_frame, text="Tipo diagramma:").pack(anchor="w")
        self.diagram_type = ttk.Combobox(control_frame, values=["Reale","Asintotico"], state="readonly")
        self.diagram_type.set("Reale")
        self.diagram_type.pack(fill="x", pady=5)
        self.diagram_type.bind("<<ComboboxSelected>>", lambda e: self.toggle_points())

        self.show_both = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Mostra reale + asintotico", variable=self.show_both).pack(anchor="w", pady=5)

        self.auto_range = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Range automatico da F(s)", variable=self.auto_range).pack(anchor="w", pady=5)

        ttk.Label(control_frame, text="Esponente pulsazione iniziale [rad/s]:").pack(anchor="w")
        self.sfe_entry = ttk.Entry(control_frame)
        self.sfe_entry.insert(0,"0")
        self.sfe_entry.pack(fill="x", pady=5)

        ttk.Label(control_frame, text="Esponente pulsazione finale [rad/s]:").pack(anchor="w")
        self.efe_entry = ttk.Entry(control_frame)
        self.efe_entry.insert(0,"9")
        self.efe_entry.pack(fill="x", pady=5)

        ttk.Label(control_frame, text="Numero punti:").pack(anchor="w")
        self.n_entry = ttk.Entry(control_frame)
        self.n_entry.insert(0,"100")
        self.n_entry.pack(fill="x", pady=5)

        ttk.Label(control_frame, text="Scala verticale modulo [min,max]:").pack(anchor="w")
        self.mag_min_entry = ttk.Entry(control_frame)
        self.mag_min_entry.insert(0,"")
        self.mag_min_entry.pack(fill="x", pady=2)
        self.mag_max_entry = ttk.Entry(control_frame)
        self.mag_max_entry.insert(0,"")
        self.mag_max_entry.pack(fill="x", pady=2)

        ttk.Label(control_frame, text="Scala verticale fase [min,max]:").pack(anchor="w")
        self.phase_min_entry = ttk.Entry(control_frame)
        self.phase_min_entry.insert(0,"")
        self.phase_min_entry.pack(fill="x", pady=2)
        self.phase_max_entry = ttk.Entry(control_frame)
        self.phase_max_entry.insert(0,"")
        self.phase_max_entry.pack(fill="x", pady=2)

        ttk.Button(control_frame, text="Genera diagramma", command=self.plot_bode).pack(fill="x", pady=5)
        ttk.Button(control_frame, text="Esporta grafico", command=self.export_graph).pack(fill="x", pady=5)

        ttk.Label(polezero_frame, text="Poli e Zeri").pack(anchor="center")
        self.pz_text = tk.Text(polezero_frame, width=30, height=40)
        self.pz_text.pack(fill="y", pady=5)

        self.fig = Figure(figsize=(10,6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_bode(self):
        try:
            expr = self.tf_entry.get().strip()
            s = sp.symbols('s')

            num_expr, den_expr = sp.fraction(sp.simplify(sp.sympify(expr)))
            num = [float(c) for c in sp.Poly(num_expr,s).all_coeffs()]
            den = [float(c) for c in sp.Poly(den_expr,s).all_coeffs()]

            tf = signal.TransferFunction(num, den)

            # numero punti
            if self.diagram_type.get()=="Reale" or self.show_both.get():
                n = int(self.n_entry.get())
            else:
                n = 1000

            # range frequenza
            if self.auto_range.get():
                min_w,max_w = self.auto_frequency_range(num,den)
            else:
                min_w = 10**int(self.sfe_entry.get())
                max_w = 10**int(self.efe_entry.get())

            w = np.logspace(np.log10(min_w), np.log10(max_w), num=n)

            wr_real, mag_real, phase_real = signal.bode(tf,w)
            mag_asym, phase_asym = self.bode_asymptotic(num,den,w)

            # allineamento asintotico
            offset_mag = mag_real[0]-mag_asym[0]
            mag_asym += offset_mag
            offset_phase = phase_real[0]-phase_asym[0]
            phase_asym += offset_phase

            self.ax1.clear()
            self.ax2.clear()

            if self.show_both.get():
                self.ax1.semilogx(w, mag_real,color='blue',linewidth=2,label='Reale')
                self.ax1.semilogx(w, mag_asym,color='red',linewidth=2,label='Asintotico')
                self.ax2.semilogx(w, phase_real,color='blue',linewidth=2,label='Reale')
                self.ax2.semilogx(w, phase_asym,color='red',linewidth=2,label='Asintotico')
                self.ax1.legend()
                self.ax2.legend()
            else:
                if self.diagram_type.get()=='Reale':
                    self.ax1.semilogx(w, mag_real,color='blue',linewidth=2)
                    self.ax2.semilogx(w, phase_real,color='blue',linewidth=2)
                else:
                    self.ax1.semilogx(w, mag_asym,color='red',linewidth=2)
                    self.ax2.semilogx(w, phase_asym,color='red',linewidth=2)

            # aggiorna lista poli/zeri
            zeros = np.roots(num)
            poles = np.roots(den)
            self.pz_text.delete(1.0,tk.END)
            z_count = Counter([round(z.real,5) for z in zeros])
            p_count = Counter([round(p.real,5) for p in poles])
            self.pz_text.insert(tk.END,"Zeri:\n")
            for val,cnt in z_count.items():
                self.pz_text.insert(tk.END,f"{val} (molteplicità {cnt})\n")
            self.pz_text.insert(tk.END,"\nPoli:\n")
            for val,cnt in p_count.items():
                self.pz_text.insert(tk.END,f"{val} (molteplicità {cnt})\n")

            # scala verticale
            try:
                mag_min = float(self.mag_min_entry.get())
                mag_max = float(self.mag_max_entry.get())
            except:
                mag_min, mag_max = mag_real.min()-10, mag_real.max()+10

            try:
                phase_min = float(self.phase_min_entry.get())
                phase_max = float(self.phase_max_entry.get())
            except:
                phase_min, phase_max = phase_real.min()-20, phase_real.max()+20

            self.ax1.set_ylim(mag_min, mag_max)
            self.ax2.set_ylim(phase_min, phase_max)

            self.ax1.set_xscale('log')
            self.ax2.set_xscale('log')
            self.ax1.set_xlim(min_w,max_w)
            self.ax2.set_xlim(min_w,max_w)
            self.ax1.set_ylabel('Modulo [dB]')
            self.ax2.set_xlabel('Pulsazione [rad/s]')
            self.ax2.set_ylabel('Fase [deg]')
            self.ax1.grid(True,which='both',linestyle='--',linewidth=0.5)
            self.ax2.grid(True,which='both',linestyle='--',linewidth=0.5)
            self.ax1.minorticks_on()
            self.ax2.minorticks_on()

            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Errore",str(e))

    def export_graph(self):
        file_path = filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('PNG files','*.png'),('PDF files','*.pdf')])
        if file_path:
            self.fig.savefig(file_path,dpi=600)

if __name__=='__main__':
    root=tk.Tk()
    app=BodeApp(root)
    root.mainloop()