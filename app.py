# =============================================================================
# APLICACIÓN WEB – ESTIMACIÓN PROBABILÍSTICA DE CAPACIDAD PORTANTE
# Universidad Nacional del Altiplano (UNAP) – Zona SW de Puno
# Versión Web 1.0 – Basada en estimacion_capacidad_portante_v3_1.py
# Framework: Streamlit
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import io
import warnings
from datetime import datetime
import zipfile

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
N_ITER        = 10_000
MPa_TO_KGCM2  = 10.1972

st.set_page_config(
    page_title="Capacidad Portante en Roca – Zona SW Puno",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# LITOLOGÍA
# ─────────────────────────────────────────────────────────────────────────────
LITOLOGIA = {
    "1": {
        "nombre"  : "Andesita",
        "mi"      : 25,
        "gamma"   : 2.54,
        "UCS_ref" : 1044.60,
        "GSI_ref" : "44 – 48",
        "color"   : "#8E44AD",
        "Fp_reg"  : (0.3654, 16.6538),
        "Fm_nota" : "5.00  (σci=102.4 MPa ≥ 100 MPa → tope)",
    },
    "2": {
        "nombre"  : "Brecha Volcánica",
        "mi"      : 17,
        "gamma"   : 2.39,
        "UCS_ref" : 552.28,
        "GSI_ref" : "37 – 43",
        "color"   : "#D35400",
        "Fp_reg"  : (0.3393, 7.2857),
        "Fm_nota" : "2.90  (σci=54.2 MPa → interpolación)",
    },
    "3": {
        "nombre"  : "Arenisca Arcósica",
        "mi"      : 16,
        "gamma"   : 2.33,
        "UCS_ref" : 602.65,
        "GSI_ref" : "43 – 52",
        "color"   : "#1ABC9C",
        "Fp_reg"  : (0.4144, 4.8616),
        "Fm_nota" : "3.13  (σci=59.1 MPa → interpolación)",
    },
}

COLORES = {
    "Carter & Kulhawy" : "#E74C3C",
    "Merifield et al." : "#2980B9",
    "Serrano et al."   : "#27AE60",
}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE CÁLCULO (idénticas al script original)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_Fp(GSI_val, tipo_roca):
    RMR  = GSI_val + 5.0
    slope, intercept = LITOLOGIA[tipo_roca]["Fp_reg"]
    return max(slope * RMR + intercept, 5.0)

def calcular_Fm(UCS_kgcm2):
    UCS_MPa = UCS_kgcm2 / MPa_TO_KGCM2
    if UCS_MPa >= 100.0:
        return 5.0
    elif UCS_MPa <= 12.5:
        return 1.0
    else:
        return 1.0 + (UCS_MPa - 12.5) / 87.5 * 4.0

def hb_params(GSI, mi, D=0.0):
    mb = mi  * np.exp((GSI - 100.0) / (28.0 - 14.0 * D))
    s  = np.exp((GSI - 100.0) / (9.0  -  3.0 * D))
    a  = 0.5 + (1.0/6.0) * (np.exp(-GSI / 15.0) - np.exp(-20.0 / 3.0))
    return mb, s, a

def carter_kulhawy(UCS, GSI, mi, gamma, D=0.0):
    mb, s, a = hb_params(GSI, mi, D)
    sa = s**a
    return max(UCS * (sa + (mb*sa + s)**a), 0.0)

def merifield(UCS, GSI, mi, gamma, D=0.0):
    mb, s, a = hb_params(GSI, mi, D)
    sa     = s**a
    Ns0_CK = sa + (mb*sa + s)**a
    ratio  = 3.5741 - 0.0285*GSI + 0.0678*mi
    return max(UCS * Ns0_CK * ratio, 0.0)

def _I(p):
    p = np.clip(p, 1e-7, np.pi/2 - 1e-7)
    return 0.5*(np.cos(p)/np.sin(p) + np.log(np.abs(np.cos(p/2)/np.sin(p/2)) + 1e-300))

def _inv_I(Iv):
    if Iv <= 0: return np.pi/2 - 1e-7
    x = max(2.0*Iv, 0.01)
    for _ in range(5000):
        xn = 2.0*Iv - np.log(x + np.sqrt(1.0 + x*x))
        if abs(xn - x) < 1e-14: x = xn; break
        x = xn
    return np.arctan(1.0 / max(x, 1e-12))

def serrano(UCS, GSI, mi, gamma, Df_m=1.50, D=0.0):
    mb, s, a = hb_params(GSI, mi, D)
    Aa = mb*(1.0-a) / (2.0**(1.0/(1.0-a)))
    if Aa < 1e-15: return 0.0
    ba  = Aa * UCS
    za  = s / (mb*Aa + 1e-300)
    q   = gamma * Df_m * 0.1
    s01 = q/ba + za
    coeffs = [2.0*s01, 1.0, 0.0, -1.0]
    roots  = np.roots(coeffs)
    cands  = [r.real for r in roots if abs(r.imag)<1e-8 and 0.005<r.real<0.9999]
    if not cands: return 0.0
    p1  = np.arcsin(float(cands[0]))
    p2  = _inv_I(_I(p1) + np.pi/2)
    sp2 = np.sin(p2); cp2 = np.cos(p2)
    if sp2 < 1e-10: return 0.0
    Nb  = (cp2/sp2)**2 / (2.0*sp2) * 1.08
    return max(ba*(Nb - za), 0.0)

def simular(UCS, GSI_media, GSI_std, mi, gamma, Df, B, D, clave_roca):
    np.random.seed(42)
    GSI_s = np.clip(np.random.normal(GSI_media, GSI_std, N_ITER), 10.0, 85.0)
    qu_CK = np.array([carter_kulhawy(UCS, g, mi, gamma, D)  for g in GSI_s])
    qu_M  = np.array([merifield(UCS, g, mi, gamma, D)        for g in GSI_s])
    qu_S  = np.array([serrano(UCS, g, mi, gamma, Df, D)      for g in GSI_s])
    Fp_arr = np.array([calcular_Fp(g, clave_roca)            for g in GSI_s])
    Fm_val = calcular_Fm(UCS)
    F_arr  = Fp_arr * Fm_val
    qadm_CK = qu_CK / F_arr
    qadm_M  = qu_M  / F_arr
    qadm_S  = qu_S  / F_arr
    return GSI_s, qu_CK, qu_M, qu_S, Fp_arr, Fm_val, F_arr, qadm_CK, qadm_M, qadm_S

def estadisticos(arr, nombre):
    p = np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "Variable"     : nombre,
        "Media"        : np.mean(arr),
        "Mediana (P50)": np.median(arr),
        "Std"          : np.std(arr),
        "CV (%)"       : np.std(arr)/np.mean(arr)*100 if np.mean(arr) > 0 else 0,
        "Mín"          : np.min(arr),
        "Máx"          : np.max(arr),
        "P01": p[0], "P05": p[1], "P10": p[2], "P25": p[3], "P50": p[4],
        "P75": p[5], "P90": p[6], "P95": p[7], "P99": p[8],
    }

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS – devuelven Figure (no guardan en disco)
# ─────────────────────────────────────────────────────────────────────────────

def _header(params, Fp_media, Fm_val, F_media):
    return (
        f"Punto: {params['__nombre']}  |  {params['tipo_roca']}  |  "
        f"GSI={params['GSI_media']:.1f}±{params['GSI_std']:.1f}  |  "
        f"UCS={params['UCS']:.1f} kg/cm²  |  mi={params['mi']}  |  "
        f"γ={params['gamma']:.2f} g/cm³\n"
        f"Fp={Fp_media:.2f}  Fm={Fm_val:.3f}  F={F_media:.2f}  "
        f"(Serrano & Olalla, 1996)  –  Df={params['Df']:.2f} m  B={params['B']:.2f} m"
    )

def g01_hist_qu_ck(qu_CK, params, Fp_m, Fm, F_m):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(_header(params, Fp_m, Fm, F_m), fontsize=8, y=1.01)
    ax.hist(qu_CK, bins=60, color=COLORES["Carter & Kulhawy"], alpha=0.82, edgecolor='white', lw=0.3)
    for pct, ls in [(5,'--'),(50,'-'),(95,':')]:
        v = np.percentile(qu_CK, pct)
        ax.axvline(v, color='black', ls=ls, lw=1.2, label=f'P{pct:02d} = {v:.0f}')
    ax.set_title('Gráfico 01 – Histograma qu  |  Carter & Kulhawy', fontsize=10,
                 fontweight='bold', color=COLORES["Carter & Kulhawy"])
    ax.set_xlabel('qu (kg/cm²)', fontsize=9); ax.set_ylabel('Frecuencia', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def g02_hist_qu_mer(qu_M, params, Fp_m, Fm, F_m):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(_header(params, Fp_m, Fm, F_m), fontsize=8, y=1.01)
    ax.hist(qu_M, bins=60, color=COLORES["Merifield et al."], alpha=0.82, edgecolor='white', lw=0.3)
    for pct, ls in [(5,'--'),(50,'-'),(95,':')]:
        v = np.percentile(qu_M, pct)
        ax.axvline(v, color='black', ls=ls, lw=1.2, label=f'P{pct:02d} = {v:.0f}')
    ax.set_title('Gráfico 02 – Histograma qu  |  Merifield et al.', fontsize=10,
                 fontweight='bold', color=COLORES["Merifield et al."])
    ax.set_xlabel('qu (kg/cm²)', fontsize=9); ax.set_ylabel('Frecuencia', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def g03_hist_qu_ser(qu_S, params, Fp_m, Fm, F_m):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(_header(params, Fp_m, Fm, F_m), fontsize=8, y=1.01)
    ax.hist(qu_S, bins=60, color=COLORES["Serrano et al."], alpha=0.82, edgecolor='white', lw=0.3)
    for pct, ls in [(5,'--'),(50,'-'),(95,':')]:
        v = np.percentile(qu_S, pct)
        ax.axvline(v, color='black', ls=ls, lw=1.2, label=f'P{pct:02d} = {v:.0f}')
    ax.set_title('Gráfico 03 – Histograma qu  |  Serrano et al.', fontsize=10,
                 fontweight='bold', color=COLORES["Serrano et al."])
    ax.set_xlabel('qu (kg/cm²)', fontsize=9); ax.set_ylabel('Frecuencia', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def g04_hist_qadm_ck(qadm_CK, params, Fp_m, Fm, F_m):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(_header(params, Fp_m, Fm, F_m), fontsize=8, y=1.01)
    ax.hist(qadm_CK, bins=60, color=COLORES["Carter & Kulhawy"], alpha=0.82, edgecolor='white', lw=0.3)
    for pct, ls in [(5,'--'),(50,'-'),(95,':')]:
        v = np.percentile(qadm_CK, pct)
        ax.axvline(v, color='black', ls=ls, lw=1.2, label=f'P{pct:02d} = {v:.2f}')
    ax.set_title('Gráfico 04 – Histograma qadm  |  Carter & Kulhawy', fontsize=10,
                 fontweight='bold', color=COLORES["Carter & Kulhawy"])
    ax.set_xlabel('qadm = qu / (Fp×Fm)  [kg/cm²]', fontsize=9)
    ax.set_ylabel('Frecuencia', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def g05_hist_qadm_mer(qadm_M, params, Fp_m, Fm, F_m):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(_header(params, Fp_m, Fm, F_m), fontsize=8, y=1.01)
    ax.hist(qadm_M, bins=60, color=COLORES["Merifield et al."], alpha=0.82, edgecolor='white', lw=0.3)
    for pct, ls in [(5,'--'),(50,'-'),(95,':')]:
        v = np.percentile(qadm_M, pct)
        ax.axvline(v, color='black', ls=ls, lw=1.2, label=f'P{pct:02d} = {v:.2f}')
    ax.set_title('Gráfico 05 – Histograma qadm  |  Merifield et al.', fontsize=10,
                 fontweight='bold', color=COLORES["Merifield et al."])
    ax.set_xlabel('qadm = qu / (Fp×Fm)  [kg/cm²]', fontsize=9)
    ax.set_ylabel('Frecuencia', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def g06_hist_qadm_ser(qadm_S, params, Fp_m, Fm, F_m):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(_header(params, Fp_m, Fm, F_m), fontsize=8, y=1.01)
    ax.hist(qadm_S, bins=60, color=COLORES["Serrano et al."], alpha=0.82, edgecolor='white', lw=0.3)
    for pct, ls in [(5,'--'),(50,'-'),(95,':')]:
        v = np.percentile(qadm_S, pct)
        ax.axvline(v, color='black', ls=ls, lw=1.2, label=f'P{pct:02d} = {v:.2f}')
    ax.set_title('Gráfico 06 – Histograma qadm  |  Serrano et al.', fontsize=10,
                 fontweight='bold', color=COLORES["Serrano et al."])
    ax.set_xlabel('qadm = qu / (Fp×Fm)  [kg/cm²]', fontsize=9)
    ax.set_ylabel('Frecuencia', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def g07_boxplot_qu(qu_CK, qu_M, qu_S, params, Fp_m, Fm, F_m):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(_header(params, Fp_m, Fm, F_m), fontsize=8, y=1.01)
    bp = ax.boxplot([qu_CK, qu_M, qu_S],
                    labels=['Carter &\nKulhawy', 'Merifield\net al.', 'Serrano\net al.'],
                    patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2.0))
    for patch, col in zip(bp['boxes'], [COLORES["Carter & Kulhawy"],
                                         COLORES["Merifield et al."],
                                         COLORES["Serrano et al."]]):
        patch.set_facecolor(col); patch.set_alpha(0.72)
    ax.set_title('Gráfico 07 – Boxplot comparativo  |  qu (kg/cm²)',
                 fontsize=10, fontweight='bold')
    ax.set_ylabel('qu (kg/cm²)', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    plt.tight_layout()
    return fig

def g08_cdf_qu(qu_CK, qu_M, qu_S, params, Fp_m, Fm, F_m):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(_header(params, Fp_m, Fm, F_m), fontsize=8, y=1.01)
    for arr, lbl in [(qu_CK,"Carter & Kulhawy"),(qu_M,"Merifield et al."),(qu_S,"Serrano et al.")]:
        vals = np.sort(arr)
        ax.plot(vals, np.arange(1, len(vals)+1)/len(vals), color=COLORES[lbl], lw=2, label=lbl)
    ax.axhline(0.05, color='gray', ls=':', lw=1, label='P05 / P95')
    ax.axhline(0.50, color='black', ls='--', lw=1, label='P50')
    ax.axhline(0.95, color='gray', ls=':', lw=1)
    ax.set_title('Gráfico 08 – CDF  |  qu (kg/cm²)', fontsize=10, fontweight='bold')
    ax.set_xlabel('qu (kg/cm²)', fontsize=9); ax.set_ylabel('Prob. acumulada', fontsize=9)
    ax.legend(fontsize=8); ax.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def g09_dist_fp_f(Fp_arr, Fm_val, F_arr, params, Fp_m, F_m):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(_header(params, Fp_m, Fm_val, F_m), fontsize=8, y=1.01)
    col = params['color_roca']
    ax.hist(Fp_arr, bins=50, color=col,      alpha=0.65, edgecolor='white', label='Fp')
    ax.hist(F_arr,  bins=50, color='#2C3E50', alpha=0.40, edgecolor='white', label='F = Fp×Fm')
    ax.axvline(Fp_m, color=col,       ls='-',  lw=1.6, label=f'Fp media = {Fp_m:.2f}')
    ax.axvline(F_m,  color='#2C3E50', ls='--', lw=1.6, label=f'F media  = {F_m:.2f}')
    ax.set_title(f'Gráfico 09 – Distribución Fp y F=Fp×Fm  (Fm={Fm_val:.3f})',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Valor del factor', fontsize=9); ax.set_ylabel('Frecuencia', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def g10_violin(qadm_CK, qadm_M, qadm_S, params, Fp_m, Fm, F_m):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(_header(params, Fp_m, Fm, F_m), fontsize=8, y=1.01)
    parts = ax.violinplot([qadm_CK, qadm_M, qadm_S],
                          positions=[1, 2, 3],
                          showmedians=True, showextrema=True, showmeans=False)
    colores_lista = [COLORES["Carter & Kulhawy"], COLORES["Merifield et al."], COLORES["Serrano et al."]]
    for pc, col in zip(parts['bodies'], colores_lista):
        pc.set_facecolor(col); pc.set_alpha(0.68); pc.set_edgecolor('white')
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_color('#555555');  parts['cbars'].set_linewidth(1)
    parts['cmins'].set_color('#555555');  parts['cmins'].set_linewidth(1)
    parts['cmaxes'].set_color('#555555'); parts['cmaxes'].set_linewidth(1)
    for i, arr in enumerate([qadm_CK, qadm_M, qadm_S], 1):
        for pct, mk in [(5,'v'),(25,'_'),(50,'o'),(75,'_'),(95,'^')]:
            ax.scatter(i, np.percentile(arr, pct), marker=mk, color='black', s=40, zorder=5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Carter &\nKulhawy', 'Merifield\net al.', 'Serrano\net al.'], fontsize=9)
    ax.set_title('Gráfico 10 – Violin plots  |  qadm = qu / (Fp×Fm)  [kg/cm²]',
                 fontsize=10, fontweight='bold')
    ax.set_ylabel('qadm (kg/cm²)', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    from matplotlib.lines import Line2D
    leyenda = [Line2D([0],[0], marker=m, color='black', ls='none', markersize=6, label=l)
               for m, l in [('v','P05'),('_','P25 / P75'),('o','P50 (mediana'),('^','P95')]]
    ax.legend(handles=leyenda, fontsize=8, loc='upper right')
    plt.tight_layout()
    return fig

def g11_percentiles(qu_CK, qu_M, qu_S, qadm_CK, qadm_M, qadm_S, params, Fp_m, Fm, F_m):
    pcts  = [5, 10, 25, 50, 75, 90, 95]
    xlbl  = [f'P{p:02d}' for p in pcts]
    x     = np.arange(len(pcts))
    w     = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(_header(params, Fp_m, Fm, F_m), fontsize=8, y=1.01)
    for ax, arrs, titulo_ax, unidad in [
        (axes[0],
         [(qu_CK,"Carter & Kulhawy"),(qu_M,"Merifield et al."),(qu_S,"Serrano et al.")],
         'qu (kg/cm²)', 'qu (kg/cm²)'),
        (axes[1],
         [(qadm_CK,"Carter & Kulhawy"),(qadm_M,"Merifield et al."),(qadm_S,"Serrano et al.")],
         'qadm = qu / (Fp×Fm)  [kg/cm²]', 'qadm (kg/cm²)'),
    ]:
        for k, (arr, lbl) in enumerate(arrs):
            vals = [np.percentile(arr, p) for p in pcts]
            bars = ax.bar(x + (k-1)*w, vals, w, label=lbl, color=COLORES[lbl], alpha=0.82, edgecolor='white')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005*max(vals),
                        f'{v:.1f}', ha='center', va='bottom', fontsize=6.5, rotation=70)
        ax.set_xticks(x); ax.set_xticklabels(xlbl, fontsize=9)
        ax.set_ylabel(unidad, fontsize=9)
        ax.set_title(titulo_ax, fontsize=9, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.suptitle('Gráfico 11 – Percentiles por método  |  qu y qadm (kg/cm²)',
                 fontsize=10, fontweight='bold', y=1.03)
    plt.tight_layout()
    return fig

def g12_cdf_qadm(qadm_CK, qadm_M, qadm_S, params, Fp_m, Fm, F_m):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(_header(params, Fp_m, Fm, F_m), fontsize=8, y=1.01)
    for arr, lbl in [(qadm_CK,"Carter & Kulhawy"),(qadm_M,"Merifield et al."),(qadm_S,"Serrano et al.")]:
        vals = np.sort(arr)
        cdf  = np.arange(1, len(vals)+1) / len(vals)
        ax.plot(vals, cdf, color=COLORES[lbl], lw=2.2, label=lbl)
        for pct in [5, 50, 95]:
            v = np.percentile(arr, pct)
            ax.axvline(v, color=COLORES[lbl], ls=':', lw=0.9, alpha=0.6)
    ax.axhline(0.05, color='gray', ls='--', lw=1, alpha=0.8, label='P05 / P95')
    ax.axhline(0.50, color='black', ls='--', lw=1.2, label='P50')
    ax.axhline(0.95, color='gray', ls='--', lw=1, alpha=0.8)
    ax.set_title('Gráfico 12 – CDF capacidad portante admisible  |  qadm (kg/cm²)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('qadm = qu / (Fp×Fm)  [kg/cm²]', fontsize=9)
    ax.set_ylabel('Probabilidad acumulada', fontsize=9)
    ax.legend(fontsize=9); ax.grid(linestyle='--', alpha=0.3)
    for arr, lbl in [(qadm_CK,"CK"),(qadm_M,"Mer."),(qadm_S,"Ser.")]:
        for pct in [5, 50, 95]:
            v = np.percentile(arr, pct)
            ax.annotate(f'{v:.2f}', xy=(v, pct/100), xytext=(4, 4),
                        textcoords='offset points', fontsize=6.5, alpha=0.8)
    plt.tight_layout()
    return fig

def g_tabla_resumen(qu_CK, qu_M, qu_S, qadm_CK, qadm_M, qadm_S,
                    Fp_arr, Fm_val, F_arr, params, Fp_m, F_m):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    fig.suptitle(
        f"Tabla de Resumen Estadístico – Punto: {params['__nombre']}  |  "
        f"{params['tipo_roca']}  |  GSI={params['GSI_media']:.1f}±{params['GSI_std']:.1f}  |  "
        f"UCS={params['UCS']:.1f} kg/cm²  |  Fp={Fp_m:.2f}  Fm={Fm_val:.3f}  F={F_m:.2f}",
        fontsize=9, fontweight='bold', y=0.98
    )
    variables = [
        ('qu  Carter & Kulhawy',   qu_CK),
        ('qu  Merifield et al.',   qu_M),
        ('qu  Serrano et al.',     qu_S),
        ('qadm Carter & Kulhawy',  qadm_CK),
        ('qadm Merifield et al.',  qadm_M),
        ('qadm Serrano et al.',    qadm_S),
        ('Fp  (factor estadíst.)', Fp_arr),
        ('F = Fp × Fm',           F_arr),
    ]
    col_headers = ['Variable', 'Unidad', 'Media', 'Mediana\n(P50)',
                   'Std', 'CV (%)', 'Mín', 'P05', 'P25', 'P50',
                   'P75', 'P95', 'Máx']
    filas = []
    for nombre, arr in variables:
        unidad = '(adim.)' if 'Fp' in nombre or 'F =' in nombre else 'kg/cm²'
        p = np.percentile(arr, [5,25,50,75,95])
        cv = np.std(arr)/np.mean(arr)*100 if np.mean(arr)>0 else 0
        filas.append([
            nombre, unidad,
            f'{np.mean(arr):.3f}', f'{np.median(arr):.3f}',
            f'{np.std(arr):.3f}',  f'{cv:.1f}%',
            f'{np.min(arr):.3f}',
            f'{p[0]:.3f}', f'{p[1]:.3f}', f'{p[2]:.3f}',
            f'{p[3]:.3f}', f'{p[4]:.3f}',
            f'{np.max(arr):.3f}',
        ])
    tbl = ax.table(cellText=filas, colLabels=col_headers,
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 0.92])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    header_color = '#2C3E50'
    col_qu_ck    = '#FDECEA'; col_qu_m  = '#E8F4FD'; col_qu_s  = '#EAF7EA'
    col_qadm_ck  = '#FDE8E8'; col_qadm_m= '#D6EAF8'; col_qadm_s= '#D5F5E3'
    col_f        = '#FEF9E7'
    row_palette = [col_qu_ck, col_qu_m, col_qu_s,
                   col_qadm_ck, col_qadm_m, col_qadm_s, col_f, col_f]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#CCCCCC'); cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor(row_palette[r-1])
    plt.tight_layout()
    return fig

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
# EXPORTAR EXCEL (en memoria)
# ─────────────────────────────────────────────────────────────────────────────

def exportar_excel_bytes(GSI_s, qu_CK, qu_M, qu_S,
                         Fp_arr, Fm_val, F_arr,
                         qadm_CK, qadm_M, qadm_S,
                         params, nombre_punto):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        p_rows = [
            ["Punto de estimación",               nombre_punto],
            ["Fecha de cálculo",                  datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["─── DATOS DE ENTRADA ───",           ""],
            ["Tipo de roca",                       params['tipo_roca']],
            ["UCS / σci (kg/cm²)",                 params['UCS']],
            ["UCS / σci (MPa)",                    params['UCS'] / MPa_TO_KGCM2],
            ["GSI – valor ingresado (media)",      params['GSI_media']],
            ["GSI – desv. estándar usada",         params['GSI_std']],
            ["mi – constante roca intacta",        params['mi']],
            ["γ – peso específico (g/cm³)",        params['gamma']],
            ["Df – profundidad empotramiento (m)", params['Df']],
            ["B – ancho de zapata (m)",            params['B']],
            ["D – factor de disturbancia",         params['D']],
            ["N iteraciones Monte Carlo",           N_ITER],
            ["─── FACTOR DE SEGURIDAD ───",        ""],
            ["RMR estimado (GSI + 5)",             params['GSI_media'] + 5.0],
            ["Fp (media MC)",                      float(np.mean(Fp_arr))],
            ["Fp (mín MC)",                        float(np.min(Fp_arr))],
            ["Fp (máx MC)",                        float(np.max(Fp_arr))],
            ["Fm (valor fijo – UCS determinístico)", Fm_val],
            ["F = Fp × Fm (media MC)",             float(np.mean(F_arr))],
            ["Fuente mi",                          params.get('fuente_mi', '—')],
            ["Fuente γ",                           params.get('fuente_gamma', '—')],
        ]
        pd.DataFrame(p_rows, columns=["Parámetro","Valor"]).to_excel(
            writer, sheet_name='1_Parametros_Entrada', index=False)

        est_rows = []
        for arr, lbl in [(qu_CK,"qu Carter & Kulhawy"),(qu_M,"qu Merifield et al."),
                         (qu_S,"qu Serrano et al."),
                         (qadm_CK,"qadm Carter & Kulhawy"),(qadm_M,"qadm Merifield et al."),
                         (qadm_S,"qadm Serrano et al.")]:
            est_rows.append(estadisticos(arr, lbl))
        est_rows.append(estadisticos(Fp_arr, "Fp (factor estadístico)"))
        est_rows.append(estadisticos(F_arr,  "F = Fp × Fm"))
        pd.DataFrame(est_rows).to_excel(writer, sheet_name='2_Estadisticos', index=False)

        pcts = [1,5,10,15,20,25,30,40,50,60,70,75,80,85,90,95,99]
        pct_rows = []
        for arr, lbl in [(qu_CK,"Carter & Kulhawy"),(qu_M,"Merifield et al."),(qu_S,"Serrano et al.")]:
            row = {"Método": lbl}
            for p in pcts: row[f"P{p:02d}_qu"] = np.percentile(arr, p)
            pct_rows.append(row)
        pd.DataFrame(pct_rows).to_excel(writer, sheet_name='3_Percentiles_qu', index=False)

        pct_qadm = []
        for arr, lbl in [(qadm_CK,"Carter & Kulhawy"),(qadm_M,"Merifield et al."),(qadm_S,"Serrano et al.")]:
            row = {"Método": lbl}
            for p in pcts: row[f"P{p:02d}_qadm"] = np.percentile(arr, p)
            pct_qadm.append(row)
        pd.DataFrame(pct_qadm).to_excel(writer, sheet_name='4_Percentiles_qadm', index=False)

        pct_f = [{"Variable": "Fp"}]; pct_f[0].update({f"P{p:02d}": np.percentile(Fp_arr, p) for p in pcts})
        pct_f.append({"Variable": "F=Fp×Fm"}); pct_f[-1].update({f"P{p:02d}": np.percentile(F_arr, p) for p in pcts})
        pd.DataFrame(pct_f).to_excel(writer, sheet_name='5_Factor_Seguridad_FpFm', index=False)

        n_m = min(2000, N_ITER)
        idx = np.linspace(0, N_ITER-1, n_m, dtype=int)
        pd.DataFrame({
            "Iteracion"         : idx,
            "GSI_simulado"      : GSI_s[idx],
            "RMR_estimado"      : GSI_s[idx] + 5.0,
            "Fp"                : Fp_arr[idx],
            "Fm"                : Fm_val,
            "F_total"           : F_arr[idx],
            "qu_CK (kg/cm²)"    : qu_CK[idx],
            "qu_M (kg/cm²)"     : qu_M[idx],
            "qu_S (kg/cm²)"     : qu_S[idx],
            "qadm_CK (kg/cm²)"  : qadm_CK[idx],
            "qadm_M (kg/cm²)"   : qadm_M[idx],
            "qadm_S (kg/cm²)"   : qadm_S[idx],
        }).to_excel(writer, sheet_name='6_Iteraciones_Muestra', index=False)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

# ── CABECERA ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#2C3E50,#3498DB);padding:18px 24px;border-radius:10px;margin-bottom:18px">
  <h2 style="color:white;margin:0;font-size:1.35rem">🪨 ESTIMACIÓN PROBABILÍSTICA DE CAPACIDAD PORTANTE EN ROCA</h2>
  <p style="color:#BDC3C7;margin:4px 0 0;font-size:0.85rem">
    Métodos: Carter &amp; Kulhawy (1988) · Merifield et al. (2006) · Serrano et al. (1994/2000)
  </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR – PARÁMETROS DE ENTRADA ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parámetros de Entrada")
    st.markdown("---")

    # Identificación
    st.markdown("### 📍 Identificación")
    nombre_punto = st.text_input("Nombre / código del punto", value="PUNTO_01",
                                  help="Ej: C-10, P-01, EST-A3")

    # Tipo de roca
    st.markdown("### 🪨 Tipo de Roca")
    opciones_roca = {k: v['nombre'] for k, v in LITOLOGIA.items()}
    clave_roca = st.radio(
        "Seleccione la litología:",
        options=list(opciones_roca.keys()),
        format_func=lambda k: f"{opciones_roca[k]}",
        help="Únicos tipos admitidos para la zona SW de Puno"
    )
    tipo_roca  = LITOLOGIA[clave_roca]['nombre']
    color_roca = LITOLOGIA[clave_roca]['color']
    lit = LITOLOGIA[clave_roca]
    st.caption(f"mi={lit['mi']} · γ={lit['gamma']} g/cm³ · UCS_ref={lit['UCS_ref']} kg/cm² · GSI zona: {lit['GSI_ref']}")

    # GSI
    st.markdown("### 📊 GSI – Índice Resistencia Geológica")
    GSI_media = st.number_input(
        f"GSI medido (rango 10–85, ref zona: {lit['GSI_ref']})",
        min_value=10.0, max_value=85.0, value=45.0, step=0.5,
        help="Mapeo geomecánico directo o conversión RMR → GSI ≈ RMR − 5"
    )
    gsi_std_sel = st.selectbox(
        "Incertidumbre del GSI (σ):",
        options=["σ = 2.5  (baja variabilidad)", "σ = 4.0  (media — recomendado)", 
                 "σ = 6.0  (alta variabilidad)", "Valor personalizado"],
        index=1
    )
    if "personalizado" in gsi_std_sel:
        GSI_std = st.number_input("σ personalizada (0.5–15)", min_value=0.5, max_value=15.0, value=4.0, step=0.5)
    else:
        GSI_std = float(gsi_std_sel.split("=")[1].split(" ")[1])

    # Vista previa Fp
    Fp_prev = calcular_Fp(GSI_media, clave_roca)
    st.caption(f"RMR estimado = {GSI_media+5:.1f}  →  Fp ≈ {Fp_prev:.2f}")

    # UCS
    st.markdown("### 🔬 UCS / σci – Resistencia Compresión Simple")
    UCS = st.number_input(
        f"UCS en laboratorio (kg/cm², ref={lit['UCS_ref']})",
        min_value=50.0, max_value=3000.0, value=float(lit['UCS_ref']), step=1.0,
        help="Norma: ASTM D7012 / ISRM. Si tiene MPa multiplique × 10.197"
    )
    Fm_prev = calcular_Fm(UCS)
    F_prev  = Fp_prev * Fm_prev
    st.caption(f"UCS = {UCS/MPa_TO_KGCM2:.2f} MPa · Fm = {Fm_prev:.4f} · F ≈ {F_prev:.2f}")

    # Parámetros opcionales
    st.markdown("### 🔧 Parámetros Opcionales")
    with st.expander("mi, γ, Df, B, D (click para editar)", expanded=False):
        usar_mi_def = st.checkbox(f"Usar mi por defecto ({lit['mi']} – Tabla Hoek-Brown)", value=True)
        if usar_mi_def:
            mi = lit['mi']
            fuente_mi = f"Tabla Hoek-Brown – {tipo_roca}"
        else:
            mi = int(st.number_input("mi personalizado (4–35)", min_value=4, max_value=35, value=lit['mi']))
            fuente_mi = "Medido en laboratorio (ensayo triaxial)"

        usar_gamma_def = st.checkbox(f"Usar γ por defecto ({lit['gamma']} g/cm³ – campo SW Puno)", value=True)
        if usar_gamma_def:
            gamma = lit['gamma']
            fuente_gamma = f"Datos de campo zona SW de Puno – {tipo_roca}"
        else:
            gamma = st.number_input("γ personalizado (g/cm³, 1.8–3.2)", min_value=1.8, max_value=3.2, value=lit['gamma'], step=0.01)
            fuente_gamma = "Medido en laboratorio (ASTM D7263)"

        usar_Df_def = st.checkbox("Usar Df = 1.50 m (estándar zona SW de Puno)", value=True)
        Df = 1.50 if usar_Df_def else st.number_input("Df (m, 0.5–5.0)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

        usar_B_def = st.checkbox("Usar B = 1.50 m (estándar zona SW de Puno)", value=True)
        B = 1.50 if usar_B_def else st.number_input("B (m, 0.5–10.0)", min_value=0.5, max_value=10.0, value=1.5, step=0.1)

        D_sel = st.selectbox(
            "Factor de disturbancia D (Hoek-Brown):",
            ["D = 0.0 – Sin perturbar (recomendado para cimentaciones)",
             "D = 0.5 – Perturbación moderada (excavación mecánica)",
             "D = 1.0 – Macizo muy perturbado (voladuras)"],
            index=0
        )
        D = float(D_sel.split("=")[1].split(" ")[1].replace("–",""))
    
    st.markdown("---")
    ejecutar = st.button("▶ EJECUTAR SIMULACIÓN MONTE CARLO", type="primary", use_container_width=True)

# ── CONTENIDO PRINCIPAL ──────────────────────────────────────────────────────

if not ejecutar:
    # Pantalla de bienvenida
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**① Configure los parámetros** en el panel izquierdo.\n\nIngrese GSI, UCS y tipo de roca del punto de campo.")
    with col2:
        st.info("**② Ejecute la simulación** presionando el botón ▶ en el panel izquierdo.\n\n10,000 iteraciones Monte Carlo con semilla 42.")
    with col3:
        st.info("**③ Descargue los resultados** al finalizar.\n\n13 gráficos PNG + Excel con 6 hojas de resultados.")

    st.markdown("---")
    st.markdown("""
    #### Metodología
    | Parámetro | Descripción |
    |---|---|
    | **Carter & Kulhawy (1988)** | Formulación exacta Hoek-Brown para qu |
    | **Merifield et al. (2006)** | Análisis límite numérico – ratio Ns0 |
    | **Serrano et al. (1994/2000)** | Solución analítica con función ψ |
    | **F = Fp × Fm** | Factor de seguridad Serrano & Olalla (1996) |
    | **RMR = GSI + 5** | Relación verificada en zona SW de Puno |
    | **N = 10,000** | Iteraciones Monte Carlo (seed = 42) |
    """)
    st.stop()

# ── EJECUCIÓN ────────────────────────────────────────────────────────────────
with st.spinner(f"⚙️ Ejecutando {N_ITER:,} iteraciones Monte Carlo para **{nombre_punto}**…"):
    GSI_s, qu_CK, qu_M, qu_S, Fp_arr, Fm_val, F_arr, qadm_CK, qadm_M, qadm_S = \
        simular(UCS, GSI_media, GSI_std, mi, gamma, Df, B, D, clave_roca)

params = {
    "__nombre"   : nombre_punto,
    "tipo_roca"  : tipo_roca,
    "color_roca" : color_roca,
    "UCS"        : UCS,
    "GSI_media"  : GSI_media,
    "GSI_std"    : GSI_std,
    "mi"         : mi,
    "gamma"      : gamma,
    "Df"         : Df,
    "B"          : B,
    "D"          : D,
    "fuente_mi"  : fuente_mi,
    "fuente_gamma": fuente_gamma,
}

Fp_m = float(np.mean(Fp_arr))
F_m  = float(np.mean(F_arr))

st.success(f"✅ Simulación completada — **{nombre_punto}** | {tipo_roca} | {N_ITER:,} iteraciones")

# ── TARJETAS DE RESULTADOS CLAVE ─────────────────────────────────────────────
st.markdown("### 📈 Resultados Clave")
colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("F = Fp × Fm (media MC)", f"{F_m:.2f}", help="Serrano & Olalla (1996)")
    st.caption(f"Fp={Fp_m:.3f} · Fm={Fm_val:.4f}")
with colB:
    st.metric("qu Media — Carter & Kulhawy", f"{np.mean(qu_CK):.1f} kg/cm²")
    st.caption(f"P05={np.percentile(qu_CK,5):.1f} · P95={np.percentile(qu_CK,95):.1f}")
with colC:
    st.metric("qu Media — Merifield et al.", f"{np.mean(qu_M):.1f} kg/cm²")
    st.caption(f"P05={np.percentile(qu_M,5):.1f} · P95={np.percentile(qu_M,95):.1f}")
with colD:
    st.metric("qu Media — Serrano et al.", f"{np.mean(qu_S):.1f} kg/cm²")
    st.caption(f"P05={np.percentile(qu_S,5):.1f} · P95={np.percentile(qu_S,95):.1f}")

colE, colF, colG = st.columns(3)
with colE:
    st.metric("qadm Media — Carter & Kulhawy", f"{np.mean(qadm_CK):.3f} kg/cm²")
    st.caption(f"P05={np.percentile(qadm_CK,5):.3f} · P50={np.percentile(qadm_CK,50):.3f} · P95={np.percentile(qadm_CK,95):.3f}")
with colF:
    st.metric("qadm Media — Merifield et al.", f"{np.mean(qadm_M):.3f} kg/cm²")
    st.caption(f"P05={np.percentile(qadm_M,5):.3f} · P50={np.percentile(qadm_M,50):.3f} · P95={np.percentile(qadm_M,95):.3f}")
with colG:
    st.metric("qadm Media — Serrano et al.", f"{np.mean(qadm_S):.3f} kg/cm²")
    st.caption(f"P05={np.percentile(qadm_S,5):.3f} · P50={np.percentile(qadm_S,50):.3f} · P95={np.percentile(qadm_S,95):.3f}")

# ── TABLA ESTADÍSTICA DETALLADA ───────────────────────────────────────────────
st.markdown("### 📋 Tabla Estadística Detallada")
filas_est = []
for arr, lbl, unidad in [
    (qu_CK,   "qu  Carter & Kulhawy",   "kg/cm²"),
    (qu_M,    "qu  Merifield et al.",   "kg/cm²"),
    (qu_S,    "qu  Serrano et al.",     "kg/cm²"),
    (qadm_CK, "qadm  Carter & Kulhawy","kg/cm²"),
    (qadm_M,  "qadm  Merifield et al.","kg/cm²"),
    (qadm_S,  "qadm  Serrano et al.",  "kg/cm²"),
    (Fp_arr,  "Fp  (factor estadíst.)", "adim."),
    (F_arr,   "F = Fp × Fm",           "adim."),
]:
    cv = np.std(arr)/np.mean(arr)*100 if np.mean(arr)>0 else 0
    p  = np.percentile(arr, [5,25,50,75,95])
    filas_est.append({
        "Variable": lbl, "Unidad": unidad,
        "Media": f"{np.mean(arr):.4f}", "Mediana": f"{np.median(arr):.4f}",
        "Std": f"{np.std(arr):.4f}",    "CV(%)": f"{cv:.2f}%",
        "Mín": f"{np.min(arr):.3f}",    "P05": f"{p[0]:.3f}",
        "P25": f"{p[1]:.3f}",           "P50": f"{p[2]:.3f}",
        "P75": f"{p[3]:.3f}",           "P95": f"{p[4]:.3f}",
        "Máx": f"{np.max(arr):.3f}",
    })
st.dataframe(pd.DataFrame(filas_est), use_container_width=True, hide_index=True)

# ── GRÁFICOS ────────────────────────────────────────────────────────────────
st.markdown("### 📊 Gráficos de Resultados")

tabs = st.tabs([
    "G01–G03: hist qu", "G04–G06: hist qadm", "G07–G08: Box & CDF qu",
    "G09: Fp & F", "G10: Violin qadm", "G11: Percentiles", "G12: CDF qadm",
    "Tabla Resumen"
])

figs_data = {}  # para almacenar bytes de PNGs descargables

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        f = g01_hist_qu_ck(qu_CK, params, Fp_m, Fm_val, F_m); st.pyplot(f); figs_data['01_hist_qu_CK'] = fig_to_bytes(f); plt.close(f)
    with c2:
        f = g02_hist_qu_mer(qu_M, params, Fp_m, Fm_val, F_m); st.pyplot(f); figs_data['02_hist_qu_Merifield'] = fig_to_bytes(f); plt.close(f)
    with c3:
        f = g03_hist_qu_ser(qu_S, params, Fp_m, Fm_val, F_m); st.pyplot(f); figs_data['03_hist_qu_Serrano'] = fig_to_bytes(f); plt.close(f)

with tabs[1]:
    c1, c2, c3 = st.columns(3)
    with c1:
        f = g04_hist_qadm_ck(qadm_CK, params, Fp_m, Fm_val, F_m); st.pyplot(f); figs_data['04_hist_qadm_CK'] = fig_to_bytes(f); plt.close(f)
    with c2:
        f = g05_hist_qadm_mer(qadm_M, params, Fp_m, Fm_val, F_m); st.pyplot(f); figs_data['05_hist_qadm_Merifield'] = fig_to_bytes(f); plt.close(f)
    with c3:
        f = g06_hist_qadm_ser(qadm_S, params, Fp_m, Fm_val, F_m); st.pyplot(f); figs_data['06_hist_qadm_Serrano'] = fig_to_bytes(f); plt.close(f)

with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        f = g07_boxplot_qu(qu_CK, qu_M, qu_S, params, Fp_m, Fm_val, F_m); st.pyplot(f); figs_data['07_boxplot_qu'] = fig_to_bytes(f); plt.close(f)
    with c2:
        f = g08_cdf_qu(qu_CK, qu_M, qu_S, params, Fp_m, Fm_val, F_m); st.pyplot(f); figs_data['08_CDF_qu'] = fig_to_bytes(f); plt.close(f)

with tabs[3]:
    f = g09_dist_fp_f(Fp_arr, Fm_val, F_arr, params, Fp_m, F_m); st.pyplot(f); figs_data['09_dist_Fp_F'] = fig_to_bytes(f); plt.close(f)

with tabs[4]:
    f = g10_violin(qadm_CK, qadm_M, qadm_S, params, Fp_m, Fm_val, F_m); st.pyplot(f); figs_data['10_violin_qadm'] = fig_to_bytes(f); plt.close(f)

with tabs[5]:
    f = g11_percentiles(qu_CK, qu_M, qu_S, qadm_CK, qadm_M, qadm_S, params, Fp_m, Fm_val, F_m)
    st.pyplot(f); figs_data['11_percentiles'] = fig_to_bytes(f); plt.close(f)

with tabs[6]:
    f = g12_cdf_qadm(qadm_CK, qadm_M, qadm_S, params, Fp_m, Fm_val, F_m); st.pyplot(f); figs_data['12_CDF_qadm'] = fig_to_bytes(f); plt.close(f)

with tabs[7]:
    f = g_tabla_resumen(qu_CK, qu_M, qu_S, qadm_CK, qadm_M, qadm_S,
                        Fp_arr, Fm_val, F_arr, params, Fp_m, F_m)
    st.pyplot(f); figs_data['00_TABLA_resumen'] = fig_to_bytes(f); plt.close(f)

# ── DESCARGAS ────────────────────────────────────────────────────────────────
st.markdown("### ⬇️ Descargar Resultados")
fecha_str = datetime.now().strftime("%Y%m%d_%H%M")
nombre_safe = nombre_punto.replace("/","_").replace(" ","_").replace("\\","_")

col_d1, col_d2 = st.columns(2)

with col_d1:
    xlsx_bytes = exportar_excel_bytes(
        GSI_s, qu_CK, qu_M, qu_S, Fp_arr, Fm_val, F_arr,
        qadm_CK, qadm_M, qadm_S, params, nombre_punto
    )
    st.download_button(
        label="📥 Descargar Excel (6 hojas)",
        data=xlsx_bytes,
        file_name=f"Estimacion_{nombre_safe}_{fecha_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with col_d2:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for nombre_png, datos_png in figs_data.items():
            zf.writestr(f"{nombre_png}_{nombre_safe}_{fecha_str}.png", datos_png)
    zip_buf.seek(0)
    st.download_button(
        label="📥 Descargar todos los PNGs (ZIP)",
        data=zip_buf.read(),
        file_name=f"Graficos_{nombre_safe}_{fecha_str}.zip",
        mime="application/zip",
        use_container_width=True
    )

# ── PIE DE PÁGINA ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Desarrollado por: J.A.C. - Todos los derechos reservados")
