import streamlit as st
import numpy as np
from scipy import signal
import sympy as sp
import matplotlib.pyplot as plt
from collections import Counter

st.title("Bode Generator v2.0")

# INPUT
st.subheader("Funzione di trasferimento F(s):")
expr = st.text_input("", "1/(s+1)")

st.subheader("Tipo diagramma:")
diagram_type = st.selectbox("", ["Reale", "Asintotico"])

show_both = st.checkbox("Mostra reale + asintotico")

col1, col2 = st.columns(2)
with col1:
    sfe = st.text_input("Esponente pulsazione iniziale log10(ω):", "0")
with col2:
    efe = st.text_input("Esponente pulsazione finale log10(ω):", "3")

# CURSORI
st.subheader("Valori cursori")

c1_exp = st.slider("Cursore 1 log10(ω)", float(sfe), float(efe), float(sfe))
c2_exp = st.slider("Cursore 2 log10(ω)", float(sfe), float(efe), float(sfe)+1)

def bode_asymptotic(num, den, w):
    zeros = np.roots(num)
    poles = np.roots(den)
    mag = np.zeros_like(w)
    phase = np.zeros_like(w)

    k = num[0] / den[0]
    mag += 20 * np.log10(abs(k))

    for z in zeros:
        wz = abs(z)
        if wz > 0:
            mag += np.where(w >= wz, 20*np.log10(w/wz), 0)

    for p in poles:
        wp = abs(p)
        if wp > 0:
            mag -= np.where(w >= wp, 20*np.log10(w/wp), 0)

    return mag, phase

if st.button("Genera diagramma"):
    try:
        s = sp.symbols("s")
        num_expr, den_expr = sp.fraction(sp.sympify(expr))

        num = [float(c) for c in sp.Poly(num_expr, s).all_coeffs()]
        den = [float(c) for c in sp.Poly(den_expr, s).all_coeffs()]

        tf = signal.TransferFunction(num, den)

        w = np.logspace(float(sfe), float(efe), 1000)
        _, mag_real, phase_real = signal.bode(tf, w)

        mag_asym, phase_asym = bode_asymptotic(num, den, w)

        if show_both:
            mag = mag_real
            phase = phase_real
        else:
            if diagram_type == "Reale":
                mag = mag_real
                phase = phase_real
            else:
                mag = mag_asym
                phase = phase_asym

        # cursori
        c1 = 10**c1_exp
        c2 = 10**c2_exp

        def get_values(x):
            idx = np.argmin(np.abs(w - x))
            return w[idx], w[idx]/(2*np.pi), mag[idx], phase[idx]

        v1 = get_values(c1)
        v2 = get_values(c2)

        # GRAFICO
        fig, (ax1, ax2) = plt.subplots(2, 1)

        if show_both:
            ax1.semilogx(w, mag_real, label="Reale")
            ax1.semilogx(w, mag_asym, label="Asintotico")
            ax2.semilogx(w, phase_real, label="Reale")
            ax2.semilogx(w, phase_asym, label="Asintotico")
            ax1.legend()
            ax2.legend()
        else:
            ax1.semilogx(w, mag)
            ax2.semilogx(w, phase)

        # linee cursori
        for x in [v1[0], v2[0]]:
            ax1.axvline(x)
            ax2.axvline(x)

        ax1.set_ylabel("Modulo [dB]")
        ax2.set_ylabel("Fase [°]")
        ax2.set_xlabel("Pulsazione ω [rad/s]")

        ax1.grid(True)
        ax2.grid(True)

        st.pyplot(fig)

        # POLI E ZERI
        st.subheader("Poli e Zeri")

        zeros = np.roots(num)
        poles = np.roots(den)

        z_count = Counter([complex(round(z.real, 5), round(z.imag, 5)) for z in zeros])
        p_count = Counter([complex(round(p.real, 5), round(p.imag, 5)) for p in poles])

        st.write("Zeri:")
        for val, cnt in z_count.items():
            st.write(f"{val} (molteplicità {cnt})")

        st.write("Poli:")
        for val, cnt in p_count.items():
            st.write(f"{val} (molteplicità {cnt})")

        # VALORI CURSORI
        st.subheader("Valori cursori")

        st.write(f"Cursore 1 → ω={v1[0]:.5g}, f={v1[1]:.5g} Hz, |G|={v1[2]:.3f} dB, ∠G={v1[3]:.3f}°")
        st.write(f"Cursore 2 → ω={v2[0]:.5g}, f={v2[1]:.5g} Hz, |G|={v2[2]:.3f} dB, ∠G={v2[3]:.3f}°")

        # DIFFERENZE
        st.subheader("Differenze")

        st.write(f"Δω = {abs(v2[0]-v1[0]):.5g} rad/s")
        st.write(f"Δf = {abs(v2[1]-v1[1]):.5g} Hz")
        st.write(f"Δ|G| = {v2[2]-v1[2]:.3f} dB")
        st.write(f"Δ∠G = {v2[3]-v1[3]:.3f}°")

    except Exception as e:
        st.error(str(e))
