"""Combinatorics and number theory task families.

Every generator returns (question, answer, solution, difficulty, tags) and
every answer is checked against an independent brute-force computation in
``verify`` before the task is accepted.
"""
from __future__ import annotations

import itertools
import math
import random

import sympy as sp
from sympy.ntheory.modular import crt


def lattice_paths_avoiding(rng: random.Random):
    m, n = rng.randint(5, 9), rng.randint(5, 9)
    bx, by = rng.randint(1, m - 1), rng.randint(1, n - 1)
    total = math.comb(m + n, m)
    through = math.comb(bx + by, bx) * math.comb(m - bx + n - by, m - bx)
    ans = total - through
    q = (
        f"A token moves on the integer grid from (0,0) to ({m},{n}) using unit steps "
        f"right (+1,0) or up (0,+1). The point ({bx},{by}) is blocked. "
        f"How many distinct paths never visit the blocked point?"
    )
    sol = (
        f"1. All monotone paths: C({m + n},{m}) = {total}.\n"
        f"2. Paths through ({bx},{by}): C({bx + by},{bx}) * C({m - bx + n - by},{m - bx}) = {through}.\n"
        f"3. Complement: {total} - {through} = {ans}."
    )
    diff = "medium" if m + n < 14 else "hard"

    def verify():
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for x in range(m + 1):
            for y in range(n + 1):
                if (x, y) == (bx, by):
                    continue
                if x == 0 and y == 0:
                    dp[x][y] = 1
                    continue
                dp[x][y] = (dp[x - 1][y] if x else 0) + (dp[x][y - 1] if y else 0)
        return dp[m][n]

    return q, str(ans), sol, diff, ["counting", "complement"], verify


def bounded_compositions(rng: random.Random):
    k, cap = rng.randint(3, 5), rng.randint(3, 7)
    n = rng.randint(cap + 1, k * cap - 1)
    terms = []
    ans = 0
    for j in range(k + 1):
        top = n - j * (cap + 1)
        if top < 0:
            break
        t = (-1) ** j * math.comb(k, j) * math.comb(top + k - 1, k - 1)
        terms.append(f"(-1)^{j} C({k},{j}) C({top + k - 1},{k - 1}) = {t}")
        ans += t
    q = (
        f"How many ordered {k}-tuples of integers (x_1,...,x_{k}) satisfy "
        f"x_1 + ... + x_{k} = {n} with 0 <= x_i <= {cap} for every i?"
    )
    sol = (
        "Inclusion-exclusion over the set of variables that exceed the cap "
        f"(substitute x_i -> x_i + {cap + 1}):\n  " + "\n  ".join(terms) + f"\nSum = {ans}."
    )
    diff = "hard" if k >= 5 else "medium"

    def verify():
        return sum(1 for t in itertools.product(range(cap + 1), repeat=k) if sum(t) == n)

    return q, str(ans), sol, diff, ["inclusion-exclusion", "stars-and-bars"], verify


def coprime_count(rng: random.Random):
    primes = rng.sample([2, 3, 5, 7, 11, 13], rng.randint(2, 4))
    M = math.prod(p ** rng.randint(1, 2) for p in primes)
    N = rng.randint(500, 5000)
    ps = sorted(sp.primefactors(M))
    ans = 0
    lines = []
    for r in range(len(ps) + 1):
        for S in itertools.combinations(ps, r):
            d = math.prod(S)
            ans += (-1) ** r * (N // d)
        lines.append(f"size-{r} subsets contribute {'+' if r % 2 == 0 else '-'} sum floor({N}/d)")
    q = f"How many integers n with 1 <= n <= {N} satisfy gcd(n, {M}) = 1?"
    sol = (
        f"1. The prime factors of {M} are {ps}.\n"
        "2. Inclusion-exclusion over products d of distinct prime factors:\n  "
        + "\n  ".join(lines)
        + f"\n3. Total = {ans}."
    )

    def verify():
        return sum(1 for i in range(1, N + 1) if math.gcd(i, M) == 1)

    return q, str(ans), sol, "medium", ["inclusion-exclusion", "gcd"], verify


def crt_system(rng: random.Random):
    mods = rng.sample([5, 7, 9, 11, 13, 16, 17, 19], 3)
    while any(math.gcd(a, b) != 1 for a, b in itertools.combinations(mods, 2)):
        mods = rng.sample([5, 7, 9, 11, 13, 16, 17, 19], 3)
    rems = [rng.randrange(m) for m in mods]
    x, L = crt(mods, rems)
    x, L = int(x), int(L)
    conds = ", ".join(f"x = {r} (mod {m})" for r, m in zip(rems, mods))
    q = f"Find the smallest positive integer x such that {conds}."
    ans = x if x > 0 else L
    sol = (
        f"The moduli {mods} are pairwise coprime, so by CRT there is a unique class mod {L}.\n"
        f"Combining the congruences pairwise gives x = {x} (mod {L}); the smallest positive "
        f"representative is {ans}."
    )

    def verify():
        return next(v for v in range(1, L + 1) if all(v % m == r for r, m in zip(rems, mods)))

    return q, str(ans), sol, "medium", ["crt", "modular-arithmetic"], verify


def last_digits_power_tower(rng: random.Random):
    a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 5)
    mod = 1000
    e = b**c
    ans = pow(a, e, mod)
    q = f"What are the last three digits of {a}^({b}^{c})? Give the answer as an integer between 0 and 999."
    sol = (
        f"1. The exponent is {b}^{c} = {e}.\n"
        f"2. Compute {a}^{e} mod 1000 by repeated squaring (or split mod 8 and mod 125 and use CRT).\n"
        f"3. Result: {ans}."
    )

    def verify():
        # independent route: residues mod 8 and mod 125 by naive multiplication, then CRT
        r8 = r125 = 1
        for _ in range(e % 2 + 4 if e >= 3 else e):  # a^e mod 8 is periodic with period 2 from e>=3
            r8 = r8 * a % 8
        for _ in range(e % 100 + 100 if e >= 100 else e):  # lambda(125) = 100; valid for e >= 3
            r125 = r125 * a % 125
        return int(crt([8, 125], [r8, r125])[0])

    return q, str(ans), sol, "hard", ["modular-exponentiation"], verify


GENERATORS = {
    "combinatorics/lattice_paths": lattice_paths_avoiding,
    "combinatorics/bounded_compositions": bounded_compositions,
    "number_theory/coprime_count": coprime_count,
    "number_theory/crt": crt_system,
    "number_theory/power_tower_digits": last_digits_power_tower,
}
