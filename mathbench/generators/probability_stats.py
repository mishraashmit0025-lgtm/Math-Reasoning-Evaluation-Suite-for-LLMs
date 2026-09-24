"""Probability and statistics task families with exact rational answers."""
from __future__ import annotations

import itertools
import random
from fractions import Fraction

import sympy as sp


def dice_sum(rng: random.Random):
    k, faces = rng.randint(3, 5), rng.choice([4, 6, 8])
    s = rng.randint(k + 2, k * faces - 2)
    x = sp.symbols("x")
    poly = sp.expand(sum(x**i for i in range(1, faces + 1)) ** k)
    ways = sp.Poly(poly, x).coeff_monomial(x**s)
    ans = sp.Rational(ways, faces**k)
    q = (
        f"{k} fair {faces}-sided dice (faces 1..{faces}) are rolled. What is the probability "
        f"that the sum of the faces equals exactly {s}? Give an exact fraction."
    )
    sol = (
        f"1. The generating function for one die is x + x^2 + ... + x^{faces}.\n"
        f"2. The coefficient of x^{s} in its {k}-th power is {ways}.\n"
        f"3. Probability = {ways}/{faces}^{k} = {ans}."
    )

    def verify():
        hits = sum(1 for r in itertools.product(range(1, faces + 1), repeat=k) if sum(r) == s)
        return sp.Rational(hits, faces**k)

    return q, str(ans), sol, "medium", ["generating-functions"], verify


def pattern_waiting_time(rng: random.Random):
    L = rng.randint(3, 5)
    pat = "".join(rng.choice("HT") for _ in range(L))
    # Markov chain over prefix-match states solved exactly.
    E = sp.symbols(f"E0:{L}")
    eqs = []
    for i in range(L):
        rhs = 1
        for c in "HT":
            s = pat[:i] + c
            while s and not pat.startswith(s):
                s = s[1:]
            j = len(s)
            rhs += sp.Rational(1, 2) * (0 if j == L else E[j])
        eqs.append(sp.Eq(E[i], rhs))
    ans = sp.solve(eqs, E)[E[0]]
    q = (
        f"A fair coin is flipped repeatedly. What is the expected number of flips until the "
        f"pattern {pat} first appears as consecutive outcomes?"
    )
    sol = (
        f"1. Let E_i be the expected remaining flips when the longest suffix matching a prefix of {pat} has length i.\n"
        "2. From each state, a flip moves to the KMP-failure state for the new character.\n"
        f"3. Solving the {L} linear equations gives E_0 = {ans}.\n"
        "   (Check: by Conway's correlation formula, E = sum of 2^k over k where the length-k prefix equals the length-k suffix.)"
    )

    def verify():
        return sum(2**k for k in range(1, L + 1) if pat[:k] == pat[-k:])

    return q, str(ans), sol, "hard", ["markov-chains", "expectation"], verify


def bayes_test(rng: random.Random):
    prev = Fraction(rng.choice([1, 2, 5]), rng.choice([100, 200, 1000]))
    sens = Fraction(rng.randint(85, 99), 100)
    spec = Fraction(rng.randint(90, 99), 100)
    two_tests = rng.random() < 0.5
    if two_tests:
        num = prev * sens**2
        den = num + (1 - prev) * (1 - spec) ** 2
        tail = "tests positive on two independent tests"
    else:
        num = prev * sens
        den = num + (1 - prev) * (1 - spec)
        tail = "tests positive"
    ans = num / den
    q = (
        f"A condition has prevalence {prev}. A test has sensitivity {sens} and specificity {spec}. "
        f"A randomly chosen person {tail}. What is the exact probability they have the condition?"
    )
    sol = (
        "Bayes' rule: P(D|+) = P(+|D)P(D) / [P(+|D)P(D) + P(+|not D)P(not D)].\n"
        f"Numerator = {num}; denominator = {den}; posterior = {ans}."
    )

    def verify():
        # enumerate the joint distribution explicitly
        p = Fraction(0)
        tot = Fraction(0)
        for d in (True, False):
            pd = prev if d else 1 - prev
            pos = sens if d else 1 - spec
            w = pd * (pos**2 if two_tests else pos)
            tot += w
            if d:
                p += w
        return p / tot

    return q, f"{ans.numerator}/{ans.denominator}", sol, "medium", ["bayes"], verify


def order_statistic(rng: random.Random):
    n = rng.randint(3, 7)
    k = rng.randint(1, n)
    x = sp.symbols("x")
    density = sp.factorial(n) / (sp.factorial(k - 1) * sp.factorial(n - k)) * x ** (k - 1) * (1 - x) ** (n - k)
    ans = sp.integrate(x**2 * density, (x, 0, 1))
    q = (
        f"{n} numbers are drawn independently and uniformly from [0,1]. Let Y be the "
        f"{k}-th smallest. Compute E[Y^2] exactly."
    )
    sol = (
        f"1. Y ~ Beta({k}, {n - k + 1}).\n"
        f"2. E[Y^2] = k(k+1)/((n+1)(n+2)) = {k}*{k + 1}/({n + 1}*{n + 2}).\n"
        f"3. = {ans}."
    )

    def verify():
        return sp.Rational(k * (k + 1), (n + 1) * (n + 2))

    return q, str(ans), sol, "hard", ["order-statistics", "beta-distribution"], verify


def variance_of_sum(rng: random.Random):
    n = rng.randint(5, 12)
    p = sp.Rational(1, rng.randint(2, 6))
    q_ = sp.Rational(rng.randint(1, 3), 4)
    ans = n * p * (1 - p) + 4 * n * q_ * (1 - q_)
    q = (
        f"X ~ Binomial({n}, {p}) and Z ~ Binomial({n}, {q_}) are independent. "
        f"Compute Var(X - 2Z) exactly."
    )
    sol = (
        f"Var(X - 2Z) = Var(X) + 4 Var(Z) = {n}*{p}*{1 - p} + 4*{n}*{q_}*{1 - q_} = {ans}."
    )

    def verify():
        vx = sum(sp.binomial(n, i) * p**i * (1 - p) ** (n - i) * (i - n * p) ** 2 for i in range(n + 1))
        vz = sum(sp.binomial(n, i) * q_**i * (1 - q_) ** (n - i) * (i - n * q_) ** 2 for i in range(n + 1))
        return sp.nsimplify(vx + 4 * vz)

    return q, str(ans), sol, "medium", ["variance", "independence"], verify


GENERATORS = {
    "probability/dice_sum": dice_sum,
    "probability/pattern_waiting_time": pattern_waiting_time,
    "probability/bayes": bayes_test,
    "statistics/order_statistic": order_statistic,
    "statistics/variance_of_sum": variance_of_sum,
}
