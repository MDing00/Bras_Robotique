import numpy as np
import matplotlib.pyplot as plt

# Paramètres du robot
L1, L2, L3, L4 = 4.0, 3.0, 2.0, 2.0

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


def fkine(theta1, theta2, theta3, theta4):
    """Cinématique directe complète"""

    # Conversion degrés -> radians
    t1 = np.deg2rad(theta1)
    t2 = np.deg2rad(theta2)
    t3 = np.deg2rad(theta3)
    t4 = np.deg2rad(theta4)

    T1 = matrice_dh(t1, 0, L1, 0)
    T2 = matrice_dh(t2, 0, L2, 0)
    T3 = matrice_dh(t3, 0, L3, 0)
    T4 = matrice_dh(t4, 0, L4, 0)

    # Transformation totale
    T = T1 @ T2 @ T3 @ T4

    # Position finale
    xe = T[0, 3]
    ye = T[1, 3]

    # Orientation totale
    theta_total = t1 + t2 + t3 + t4

    return T, xe, ye, theta_total


def dessiner_robot(theta1, theta2, theta3, theta4):
    """Affichage du robot"""

    T, xe, ye, theta_total = fkine(theta1, theta2, theta3, theta4)

    # Conversion
    t1 = np.deg2rad(theta1)
    t2 = np.deg2rad(theta2)
    t3 = np.deg2rad(theta3)

    # Matrices intermédiaires
    T1 = matrice_dh(t1, 0, L1, 0)
    T2 = matrice_dh(t2, 0, L2, 0)
    T3 = matrice_dh(t3, 0, L3, 0)

    T12 = T1 @ T2
    T123 = T12 @ T3

    # Positions des articulations
    x1, y1 = T1[0, 3], T1[1, 3]
    x2, y2 = T12[0, 3], T12[1, 3]
    x3, y3 = T123[0, 3], T123[1, 3]

    plt.figure(figsize=(10, 10))
    plt.axis('equal')
    plt.grid(True)

    # Tracé des segments
    plt.plot([0, x1], [0, y1], 'b-', linewidth=6)
    plt.plot([x1, x2], [y1, y2], 'g-', linewidth=6)
    plt.plot([x2, x3], [y2, y3], 'orange', linewidth=6)
    plt.plot([x3, xe], [y3, ye], 'r-', linewidth=6)

    # Points
    plt.plot([0, x1, x2, x3, xe], [0, y1, y2, y3, ye], 'ko', markersize=10)
    plt.plot(xe, ye, 'r*', markersize=15)

    plt.title(f'Configuration : θ1={theta1:.1f} | θ2={theta2:.1f} | θ3={theta3:.1f} | θ4={theta4:.1f}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


def espace_travail(n=2000):
    """Affichage de l'espace de travail"""

    X = []
    Y = []

    for i in range(n):
        th1 = np.random.uniform(-180, 180)
        th2 = np.random.uniform(-180, 180)
        th3 = np.random.uniform(-180, 180)
        th4 = np.random.uniform(-180, 180)

        _, xe, ye, _ = fkine(th1, th2, th3, th4)

        X.append(xe)
        Y.append(ye)

    plt.figure(figsize=(9, 9))
    plt.scatter(X, Y, s=3, alpha=0.5, color='blue')
    plt.axis('equal')
    plt.grid(True)
    plt.title('Espace de Travail du Bras RRRR')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


# Programme principal
if __name__ == "__main__":
    # Exemple de configuration
    dessiner_robot(90, 0, 0, 0)

    # Décommenter pour voir l’espace de travail
    # espace_travail(3000)