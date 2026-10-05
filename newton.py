"""Newton interpolation of f(x) = x^3 - 2x + 1 at x = 0, 1, 2."""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def f(x):
    return x**3 - 2*x + 1


def divided_differences(xs, ys):
    """Return the full table and the Newton coefficients f[x0], f[x0,x1], ..."""
    n = len(xs)
    table = [list(ys)]
    for k in range(1, n):
        prev = table[-1]
        table.append([(prev[i + 1] - prev[i]) / (xs[i + k] - xs[i])
                      for i in range(n - k)])
    coeffs = [col[0] for col in table]
    return table, coeffs


def newton(xs, coeffs, x):
    """Evaluate the Newton form with a Horner-like scheme."""
    result = coeffs[-1]
    for k in range(len(coeffs) - 2, -1, -1):
        result = result * (x - xs[k]) + coeffs[k]
    return result


if __name__ == "__main__":
    xs = [0, 1, 2]
    ys = [sp.Integer(f(xi)) for xi in xs]          # [1, 0, 5]

    table, coeffs = divided_differences(xs, ys)
    for k, col in enumerate(table):
        print(f"order {k}:", col)
    print("Newton coefficients:", coeffs)           # [1, -1, 3]

    X = sp.symbols("x")
    P = sp.expand(newton(xs, coeffs, X))
    print("P_N(x) =", P)                            # 3*x**2 - 4*x + 1

    P_num = sp.lambdify(X, P, "numpy")
    t = np.linspace(-0.6, 2.4, 400)

    plt.figure(figsize=(8, 5))
    plt.plot(t, f(t), "b-", lw=2, label=r"$f(x)=x^3-2x+1$")
    plt.plot(t, P_num(t), "r--", lw=2, label=r"$P_N(x)=3x^2-4x+1$")
    plt.scatter(xs, [float(y) for y in ys], c="orange", zorder=5, label="Nodes")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Newton interpolation")
    plt.legend()
    plt.savefig("newton.png", dpi=150, bbox_inches="tight")
    plt.show()
