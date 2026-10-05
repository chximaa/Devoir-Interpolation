"""Hermite interpolation of f(x) = x^3 - 2x + 1 at x = 0, 1, 2
(using f and f' at each node, Newton-Hermite divided differences)."""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def f(x):
    return x**3 - 2*x + 1


def df(x):
    return 3*x**2 - 2


def hermite_divided_differences(xs, ys, dys):
    """Divided differences with each node repeated twice: f[x_i, x_i] = f'(x_i)."""
    z = [xi for xi in xs for _ in range(2)]
    n = len(z)
    table = [[ys[i // 2] for i in range(n)]]
    for k in range(1, n):
        prev = table[-1]
        col = []
        for i in range(n - k):
            if z[i + k] == z[i]:                    # repeated node -> derivative
                col.append(dys[i // 2])
            else:
                col.append((prev[i + 1] - prev[i]) / (z[i + k] - z[i]))
        table.append(col)
    coeffs = [col[0] for col in table]
    return z, table, coeffs


def newton_hermite(z, coeffs, x):
    result = coeffs[-1]
    for k in range(len(coeffs) - 2, -1, -1):
        result = result * (x - z[k]) + coeffs[k]
    return result


if __name__ == "__main__":
    xs = [0, 1, 2]
    ys = [sp.Integer(f(xi)) for xi in xs]           # [1, 0, 5]
    dys = [sp.Integer(df(xi)) for xi in xs]         # [-2, 1, 10]

    z, table, coeffs = hermite_divided_differences(xs, ys, dys)
    print("Repeated nodes z :", z)
    for k, col in enumerate(table):
        print(f"order {k}:", col)
    print("Hermite coefficients:", coeffs)          # [1, -2, 1, 1, 0, 0]

    X = sp.symbols("x")
    H = sp.expand(newton_hermite(z, coeffs, X))
    print("H(x) =", H)                              # x**3 - 2*x + 1

    H_num = sp.lambdify(X, H, "numpy")
    t = np.linspace(-0.6, 2.4, 400)

    plt.figure(figsize=(8, 5))
    plt.plot(t, f(t), "b-", lw=3, label=r"$f(x)=x^3-2x+1$")
    plt.plot(t, H_num(t), "k:", lw=2, label=r"$H(x)=x^3-2x+1$")
    plt.scatter(xs, [float(y) for y in ys], c="orange", zorder=5, label="Nodes")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Hermite interpolation")
    plt.legend()
    plt.savefig("hermite.png", dpi=150, bbox_inches="tight")
    plt.show()
