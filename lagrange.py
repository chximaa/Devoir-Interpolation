"""Lagrange interpolation of f(x) = x^3 - 2x + 1 at x = 0, 1, 2."""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def f(x):
    return x**3 - 2*x + 1


def lagrange_basis(xs, i, x):
    """L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)."""
    L = 1
    for j, xj in enumerate(xs):
        if j != i:
            L = L * (x - xj) / (xs[i] - xj)
    return L


def lagrange(xs, ys, x):
    """P(x) = sum_i f(x_i) L_i(x)."""
    return sum(ys[i] * lagrange_basis(xs, i, x) for i in range(len(xs)))


if __name__ == "__main__":
    xs = [0, 1, 2]
    ys = [sp.Integer(f(xi)) for xi in xs]          # [1, 0, 5]

    X = sp.symbols("x")
    print("Nodes  :", xs)
    print("Values :", ys)
    for i in range(len(xs)):
        print(f"L_{i}(x) =", sp.factor(lagrange_basis(xs, i, X)))

    P = sp.expand(lagrange(xs, ys, X))
    print("P_L(x) =", P)                            # 3*x**2 - 4*x + 1

    P_num = sp.lambdify(X, P, "numpy")
    t = np.linspace(-0.6, 2.4, 400)

    plt.figure(figsize=(8, 5))
    plt.plot(t, f(t), "b-", lw=2, label=r"$f(x)=x^3-2x+1$")
    plt.plot(t, P_num(t), "r--", lw=2, label=r"$P_L(x)=3x^2-4x+1$")
    plt.scatter(xs, [float(y) for y in ys], c="orange", zorder=5, label="Nodes")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Lagrange interpolation")
    plt.legend()
    plt.savefig("lagrange.png", dpi=150, bbox_inches="tight")
    plt.show()
