"""Linear algebra, calculus, recurrences and optimization task families."""
from __future__ import annotations

import random

import mpmath
import sympy as sp


def integer_determinant(rng: random.Random):
    n = rng.randint(3, 5)
    M = sp.Matrix(n, n, lambda i, j: rng.randint(-5, 5))
    ans = M.det()
    rows = "; ".join("[" + ", ".join(str(v) for v in M.row(i)) + "]" for i in range(n))
    q = f"Compute the determinant of the {n}x{n} matrix with rows {rows}."
    sol = (
        "Row-reduce with exact rational arithmetic, tracking row swaps and scaling; "
        f"the product of the pivots (with sign) is {ans}."
    )

    def verify():
        return M.det(method="berkowitz")

    return q, str(ans), sol, "medium" if n < 5 else "hard", ["determinant"], verify


def trace_of_power(rng: random.Random):
    a, b, c = rng.randint(-3, 3), rng.randint(1, 4), rng.randint(-3, 3)
    k = rng.randint(4, 8)
    M = sp.Matrix([[a, b], [c, a + 1]])
    ans = (M**k).trace()
    q = f"Let A = [[{a}, {b}], [{c}, {a + 1}]]. Compute trace(A^{k})."
    lam = M.eigenvals()
    sol = (
        f"1. Eigenvalues of A: {list(lam)}.\n"
        f"2. trace(A^{k}) is the sum of the {k}-th powers of the eigenvalues "
        f"(or use Cayley-Hamilton t_k = tr(A) t_(k-1) - det(A) t_(k-2)).\n"
        f"3. Result: {ans}."
    )

    def verify():
        t0, t1 = 2, M.trace()
        tr, dt = M.trace(), M.det()
        for _ in range(k - 1):
            t0, t1 = t1, tr * t1 - dt * t0
        return t1

    return q, str(ans), sol, "hard", ["eigenvalues", "cayley-hamilton"], verify


def definite_integral(rng: random.Random):
    x = sp.symbols("x")
    kind = rng.choice(["poly_exp", "rational", "trig"])
    if kind == "poly_exp":
        n, a = rng.randint(1, 3), rng.randint(1, 3)
        f, lo, hi = x**n * sp.exp(-a * x), 0, sp.oo
    elif kind == "rational":
        a = rng.randint(1, 4)
        f, lo, hi = 1 / (x**2 + a**2), 0, sp.oo
        if rng.random() < 0.5:
            f, hi = x / (x**2 + a**2) ** 2, sp.oo
    else:
        m, n = rng.randint(1, 4), rng.randint(1, 4)
        f, lo, hi = sp.sin(m * x) ** 2 * sp.cos(n * x) ** 2, 0, sp.pi
    ans = sp.simplify(sp.integrate(f, (x, lo, hi)))
    hi_s = "infinity" if hi == sp.oo else str(hi)
    q = f"Evaluate the integral of {sp.sstr(f)} with respect to x from {lo} to {hi_s}. Give an exact answer."
    sol = f"Antidifferentiate (integration by parts / substitution / product-to-sum) and evaluate the limits: {ans}."

    def verify():
        g = sp.lambdify(x, f, "mpmath")
        return mpmath.quad(g, [lo, mpmath.inf if hi == sp.oo else float(hi)])

    return q, sp.sstr(ans), sol, "medium", ["integration", kind], verify


def linear_recurrence(rng: random.Random):
    c1, c2 = rng.randint(1, 4), rng.randint(-3, 3) or 1
    a0, a1 = rng.randint(0, 3), rng.randint(1, 5)
    n = rng.randint(10, 20)
    k = sp.symbols("k", integer=True)
    a = sp.Function("a")
    closed = sp.rsolve(a(k) - c1 * a(k - 1) - c2 * a(k - 2), a(k), {a(0): a0, a(1): a1})
    ans = sp.nsimplify(sp.expand(closed.subs(k, n)))
    q = (
        f"A sequence satisfies a_0 = {a0}, a_1 = {a1}, and a_k = {c1} a_(k-1) + ({c2}) a_(k-2) "
        f"for k >= 2. Compute a_{n}."
    )
    sol = (
        f"1. Characteristic polynomial r^2 - {c1} r - ({c2}) = 0.\n"
        f"2. Closed form fitted to the initial conditions: a_k = {sp.sstr(closed)}.\n"
        f"3. a_{n} = {ans}."
    )

    def verify():
        u, v = a0, a1
        for _ in range(n - 1):
            u, v = v, c1 * v + c2 * u
        return v if n >= 1 else u

    return q, str(ans), sol, "medium", ["recurrences"], verify


def constrained_quadratic(rng: random.Random):
    x, y, lam = sp.symbols("x y lam", real=True)
    a, b = rng.randint(1, 5), rng.randint(1, 5)
    p, q_, c = rng.randint(1, 4), rng.randint(1, 4), rng.randint(3, 12)
    f = a * x**2 + b * y**2
    g = p * x + q_ * y - c
    sols = sp.solve([sp.diff(f - lam * g, x), sp.diff(f - lam * g, y), g], [x, y, lam], dict=True)
    ans = sp.simplify(f.subs(sols[0]))
    q = f"Find the minimum value of {a}x^2 + {b}y^2 subject to {p}x + {q_}y = {c}."
    sol = (
        f"Lagrange multipliers: 2*{a}x = {p} lam, 2*{b}y = {q_} lam. Substituting into the constraint "
        f"gives x = {sols[0][x]}, y = {sols[0][y]}, and the minimum {ans} "
        f"(= c^2 / (p^2/a + q^2/b) by Cauchy-Schwarz)."
    )

    def verify():
        return sp.Rational(c**2) / (sp.Rational(p**2, a) + sp.Rational(q_**2, b))

    return q, str(ans), sol, "medium", ["lagrange-multipliers", "cauchy-schwarz"], verify


def power_sum_mod(rng: random.Random):
    n, pw, m = rng.randint(50, 400), rng.randint(2, 5), rng.choice([97, 101, 1009, 1000])
    k = sp.symbols("k", integer=True, positive=True)
    closed = sp.factor(sp.summation(k**pw, (k, 1, sp.Symbol("N", integer=True, positive=True))))
    total = sp.summation(k**pw, (k, 1, n))
    ans = int(total) % m
    q = f"Compute (1^{pw} + 2^{pw} + ... + {n}^{pw}) mod {m}."
    sol = f"Faulhaber: sum_(k<=N) k^{pw} = {sp.sstr(closed)}. With N = {n} the sum is {total}, and mod {m} this is {ans}."

    def verify():
        return sum(pow(i, pw, m) for i in range(1, n + 1)) % m

    return q, str(ans), sol, "medium", ["faulhaber", "modular-arithmetic"], verify


GENERATORS = {
    "linear_algebra/determinant": integer_determinant,
    "linear_algebra/trace_of_power": trace_of_power,
    "calculus/definite_integral": definite_integral,
    "algebra/linear_recurrence": linear_recurrence,
    "optimization/constrained_quadratic": constrained_quadratic,
    "algebra/power_sum_mod": power_sum_mod,
}
