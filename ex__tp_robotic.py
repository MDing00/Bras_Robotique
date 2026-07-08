"""
Simulation Interactive - Bras RRR Plan
Cinématique Directe par Convention Denavit-Hartenberg
TP Robotique Avancée - Polytech Monastir
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyArrowPatch

# ──────────────────────────────────────────────
# Paramètres géométriques du robot
# ──────────────────────────────────────────────
L1, L2, L3 = 2.0, 3.0, 2.0
L_MAX = L1 + L2 + L3  # rayon max espace de travail

# ──────────────────────────────────────────────
# Cinématique directe
# ──────────────────────────────────────────────
def matrice_DH(theta_deg, d, a, alpha_deg):
    """Matrice de transformation homogène DH 4x4."""
    t = np.deg2rad(theta_deg)
    al = np.deg2rad(alpha_deg)
    ct, st = np.cos(t), np.sin(t)
    ca, sa = np.cos(al), np.sin(al)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ])

def forward_kinematics(q):
    """Retourne la liste des positions des joints + effecteur."""
    th1, th2, th3 = np.deg2rad(q)
    x0, y0 = 0.0, 0.0
    x1 = L1 * np.cos(th1)
    y1 = L1 * np.sin(th1)
    x2 = x1 + L2 * np.cos(th1 + th2)
    y2 = y1 + L2 * np.sin(th1 + th2)
    x3 = x2 + L3 * np.cos(th1 + th2 + th3)
    y3 = y2 + L3 * np.sin(th1 + th2 + th3)
    return [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]

# ──────────────────────────────────────────────
# Espace de travail (cercle discret)
# ──────────────────────────────────────────────
def compute_workspace(n=3000):
    """Monte-Carlo : échantillonnage aléatoire de l'espace de travail."""
    xs, ys = [], []
    for _ in range(n):
        q = np.random.uniform(-np.pi, np.pi, 3)
        pts = forward_kinematics(np.degrees(q))
        xs.append(pts[-1][0])
        ys.append(pts[-1][1])
    return xs, ys

# ──────────────────────────────────────────────
# Mise en place de la figure
# ──────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9), facecolor='#F8F8F6')
fig.canvas.manager.set_window_title("Simulation Bras RRR — Cinématique Directe DH")

# Zone principale du robot
ax = fig.add_axes([0.05, 0.28, 0.60, 0.68])
ax.set_facecolor('#F0EFE8')
ax.set_xlim(-10.5, 10.5)
ax.set_ylim(-10.5, 10.5)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', linewidth=0.4, color='#CCCCBB', alpha=0.7)
ax.axhline(0, color='#AAAAAA', linewidth=0.8)
ax.axvline(0, color='#AAAAAA', linewidth=0.8)
ax.set_xlabel('X (m)', fontsize=11)
ax.set_ylabel('Y (m)', fontsize=11)
ax.set_title('Bras RRR Plan — Simulation Interactive', fontsize=13, fontweight='bold', pad=10)

# Espace de travail en arrière-plan
ws_x, ws_y = compute_workspace(4000)
ax.scatter(ws_x, ws_y, s=1, c='#4A90D9', alpha=0.06, zorder=1)

# Cercle espace de travail max
theta_c = np.linspace(0, 2*np.pi, 300)
ax.plot(L_MAX * np.cos(theta_c), L_MAX * np.sin(theta_c),
        '--', color='#E24B4A', linewidth=1.0, alpha=0.5, zorder=2, label=f'Portée max = {L_MAX} m')

# ── Éléments graphiques du robot (initialisés à q=[30,45,-30]) ──
q_init = [60.0, 90.0, 30.0]
pts = forward_kinematics(q_init)

COLORS = ['#2563EB', '#16A34A', '#D97706']  # bleu, vert, ambre
WIDTHS  = [6, 5, 4]

# Lignes des segments
lines = []
for i in range(3):
    lobj, = ax.plot([], [], '-', color=COLORS[i], linewidth=WIDTHS[i],
                    solid_capstyle='round', zorder=4)
    lines.append(lobj)

# Joints (cercles creux)
joint_circles = []
joint_sizes = [120, 90, 70]
joint_colors_edge = ['#1D4ED8', '#15803D', '#B45309']
for i in range(3):
    sc = ax.scatter([], [], s=joint_sizes[i], zorder=6,
                    facecolors='white', edgecolors=joint_colors_edge[i], linewidths=2.5)
    joint_circles.append(sc)

# Base (triangle)
base_tri = plt.Polygon([[-0.35, 0], [0.35, 0], [0, -0.55]], closed=True,
                        facecolor='#374151', edgecolor='#111827', linewidth=1.2, zorder=7)
ax.add_patch(base_tri)

# Effecteur (point rouge)
eff_dot = ax.scatter([], [], s=120, color='#E24B4A', zorder=8, marker='o')

# Texte coordonnées effecteur
coord_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                     fontsize=10, va='top', ha='left',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                               edgecolor='#DDDDDD', alpha=0.9))

# Légende des segments
patches = [
    mpatches.Patch(color=COLORS[0], label=f'L₁ = {L1} m'),
    mpatches.Patch(color=COLORS[1], label=f'L₂ = {L2} m'),
    mpatches.Patch(color=COLORS[2], label=f'L₃ = {L3} m'),
    mpatches.Patch(color='#E24B4A', label='Portée max'),
]
ax.legend(handles=patches, loc='lower left', fontsize=9,
          framealpha=0.9, edgecolor='#CCCCCC')

# ── Panneau d'infos DH (droite) ──
ax_info = fig.add_axes([0.68, 0.28, 0.30, 0.68])
ax_info.set_facecolor('#F0EFE8')
ax_info.axis('off')
ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)

info_title = ax_info.text(0.5, 0.97, 'Paramètres DH', ha='center', va='top',
                           fontsize=12, fontweight='bold')
ax_info.text(0.5, 0.91, 'Tableau de configuration', ha='center', va='top',
             fontsize=9, color='#666666')

# Tableau DH dynamique
table_header = ax_info.text(0.1, 0.84, 'i    θᵢ       dᵢ   aᵢ   αᵢ',
                              fontfamily='monospace', fontsize=9,
                              va='top', color='#333333')
ax_info.axhline(0.82, xmin=0.05, xmax=0.95, color='#AAAAAA', linewidth=0.8)

dh_texts = []
for i in range(3):
    t = ax_info.text(0.1, 0.78 - i*0.08, '', fontfamily='monospace',
                     fontsize=9, va='top', color='#111111')
    dh_texts.append(t)

ax_info.axhline(0.54, xmin=0.05, xmax=0.95, color='#AAAAAA', linewidth=0.8)

# Résultats cinématique
res_title = ax_info.text(0.5, 0.50, 'Cinématique directe', ha='center',
                          fontsize=11, fontweight='bold')
res_texts = []
labels_res = ['Pos. J₁ (x₁, y₁)', 'Pos. J₂ (x₂, y₂)', 'Effecteur (x₃, y₃)', 'Distance |OE|', 'Angle total θ₁+θ₂+θ₃']
for i, lbl in enumerate(labels_res):
    ax_info.text(0.05, 0.43 - i*0.085, lbl, fontsize=8.5, va='top', color='#666666')
    t = ax_info.text(0.05, 0.415 - i*0.085, '', fontsize=9, va='top',
                     fontweight='bold', color='#1D4ED8')
    res_texts.append(t)

# ──────────────────────────────────────────────
# Sliders (θ₁, θ₂, θ₃)
# ──────────────────────────────────────────────
slider_color = '#DBEAFE'
ax_s1 = fig.add_axes([0.10, 0.19, 0.50, 0.030], facecolor=slider_color)
ax_s2 = fig.add_axes([0.10, 0.13, 0.50, 0.030], facecolor=slider_color)
ax_s3 = fig.add_axes([0.10, 0.07, 0.50, 0.030], facecolor=slider_color)

s1 = Slider(ax_s1, 'θ₁  (°)', -180, 180, valinit=q_init[0], valstep=1,
            color='#2563EB')
s2 = Slider(ax_s2, 'θ₂  (°)', -180, 180, valinit=q_init[1], valstep=1,
            color='#16A34A')
s3 = Slider(ax_s3, 'θ₃  (°)', -180, 180, valinit=q_init[2], valstep=1,
            color='#D97706')

for sl in [s1, s2, s3]:
    sl.label.set_fontsize(10)
    sl.valtext.set_fontsize(10)

# ──────────────────────────────────────────────
# Boutons de configuration prédéfinie
# ──────────────────────────────────────────────
btn_style = dict(color='#E5E7EB', hovercolor='#D1D5DB')

ax_b0 = fig.add_axes([0.05, 0.01, 0.13, 0.040])
ax_b1 = fig.add_axes([0.20, 0.01, 0.13, 0.040])
ax_b2 = fig.add_axes([0.35, 0.01, 0.13, 0.040])
ax_b3 = fig.add_axes([0.50, 0.01, 0.13, 0.040])
ax_b4 = fig.add_axes([0.65, 0.01, 0.13, 0.040])

b0 = Button(ax_b0, 'q=[0,0,0]',    **btn_style)
b1 = Button(ax_b1, 'q=[30,45,-30]',**btn_style)
b2 = Button(ax_b2, 'q=[90,0,0]',   **btn_style)
b3 = Button(ax_b3, 'Replié',        **btn_style)
b4 = Button(ax_b4, '↺ Reset',       **btn_style)

# ──────────────────────────────────────────────
# Fonction de mise à jour
# ──────────────────────────────────────────────
def update(val=None):
    q = [s1.val, s2.val, s3.val]
    pts = forward_kinematics(q)

    # Mise à jour des segments
    for i in range(3):
        p_start = pts[i]
        p_end   = pts[i + 1]
        lines[i].set_data([p_start[0], p_end[0]], [p_start[1], p_end[1]])

    # Mise à jour des joints
    for i in range(3):
        joint_circles[i].set_offsets([pts[i]])

    # Effecteur
    eff_dot.set_offsets([pts[3]])

    # Texte coordonnées sur le graphe
    xe, ye = pts[3]
    dist = np.sqrt(xe**2 + ye**2)
    coord_text.set_text(
        f'Effecteur : ({xe:.3f} m, {ye:.3f} m)\n'
        f'Distance  : {dist:.3f} m'
    )

    # Tableau DH (panneau droite)
    a_vals = [L1, L2, L3]
    for i in range(3):
        dh_texts[i].set_text(
            f' {i+1}   {q[i]:+7.1f}°    0   {a_vals[i]:.0f}    0°'
        )

    # Résultats cinématique
    labels_vals = [
        f'({pts[1][0]:.3f},  {pts[1][1]:.3f}) m',
        f'({pts[2][0]:.3f},  {pts[2][1]:.3f}) m',
        f'({pts[3][0]:.3f},  {pts[3][1]:.3f}) m',
        f'{dist:.3f} m  ({dist/L_MAX*100:.1f}% portée)',
        f'{q[0]+q[1]+q[2]:.1f}°',
    ]
    for i, t in enumerate(res_texts):
        t.set_text(labels_vals[i])

    fig.canvas.draw_idle()

# ── Callbacks boutons ──
def set_q(q_vals):
    s1.set_val(q_vals[0])
    s2.set_val(q_vals[1])
    s3.set_val(q_vals[2])

b0.on_clicked(lambda e: set_q([0,   0,   0  ]))
b1.on_clicked(lambda e: set_q([30,  45,  -30]))
b2.on_clicked(lambda e: set_q([90,  0,   0  ]))
b3.on_clicked(lambda e: set_q([0,   90,  -90]))
b4.on_clicked(lambda e: set_q(q_init))

s1.on_changed(update)
s2.on_changed(update)
s3.on_changed(update)

# Premier rendu
update()

plt.show()



##############################################################################################
###############################################################################################
#############################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Paramètres du robot
L1 = 4.0
L2 = 3.0
L3 = 2.0

def matrice_dh(theta, d, a, alpha):
    """Retourne la matrice homogène 4x4"""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0 ,     sa,     ca,    d],
        [0 ,      0,      0,    1]
    ])
    return T


def fkine(theta1, theta2, theta3):
    """Cinématique directe complète"""

    # Conversion degrés -> radians
    t1 = np.deg2rad(theta1)
    t2 = np.deg2rad(theta2)
    t3 = np.deg2rad(theta3)

    T1 = matrice_dh(t1, 0, L1, 0)
    T2 = matrice_dh(t2, 0, L2, 0)
    T3 = matrice_dh(t3, 0, L3, 0)

    # Transformation totale
    T = T1 @ T2 @ T3

    # Position
    xe = T[0, 3]
    ye = T[1, 3]

    # Orientation
    theta_total = t1 + t2 + t3

    return T, xe, ye, theta_total


def dessiner_robot(theta1, theta2, theta3):
    T, xe, ye, theta_total = fkine(theta1, theta2, theta3)

    # Matrices intermédiaires
    t1 = np.deg2rad(theta1)
    t2 = np.deg2rad(theta2)

    T1 = matrice_dh(t1, 0, L1, 0)
    T2 = matrice_dh(t2, 0, L2, 0)

    T12 = T1 @ T2

    x1 = T1[0, 3]
    y1 = T1[1, 3]

    x2 = T12[0, 3]
    y2 = T12[1, 3]

    plt.figure(figsize=(10, 10))
    plt.axis('equal')
    plt.grid(True)

    # Tracé
    plt.plot([0, x1], [0, y1], 'b-', linewidth=6)
    plt.plot([x1, x2], [y1, y2], 'g-', linewidth=6)
    plt.plot([x2, xe], [y2, ye], 'orange', linewidth=6)

    plt.plot([0, x1, x2, xe], [0, y1, y2, ye], 'ko', markersize=10)
    plt.plot(xe, ye, 'r*', markersize=15)

    plt.title(f'Configuration : θ1={theta1:.1f} | θ2={theta2:.1f} | θ3={theta3:.1f}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


def espace_travail(n=2000):
    X = []
    Y = []

    for i in range(n):
        th1 = np.random.uniform(-180, 180)
        th2 = np.random.uniform(-180, 180)
        th3 = np.random.uniform(-180, 180)

        _, xe, ye, _ = fkine(th1, th2, th3)

        X.append(xe)
        Y.append(ye)

    plt.figure(figsize=(9, 9))
    plt.scatter(X, Y, s=3, alpha=0.5, color='blue')
    plt.axis('equal')
    plt.grid(True)
    plt.title('Espace de Travail du Bras RRR')
    plt.show()
