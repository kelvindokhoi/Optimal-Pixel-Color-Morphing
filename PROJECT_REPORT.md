# Project Report: Optimal Pixel Assignment Between Two Images

## Project question

How can every pixel from a source photograph be rearranged exactly once so the
new arrangement resembles a second target image, while discouraging excessively
long pixel movements?

The source is the supplied portrait of Maria Siopsis. The target is the Obama
portrait used by the open-source Obamify project. The final image uses Maria's
original pixel palette but arranges it according to Obama's visual structure.

## Linear-programming model

Let `P={1,...,n}` index the source pixels and target locations. On the default
80×80 grid, `n=6400`.

### Data

- `(R_i,G_i,B_i)`: RGB color of source pixel `i`;
- `(R'_j,G'_j,B'_j)`: RGB color desired at target location `j`;
- `(r_i,c_i)` and `(r_j,c_j)`: normalized source and target coordinates;
- `λ≥0`: proximity importance.

For each possible assignment, define the constant cost

```text
C_ij = (R_i-R'_j)² + (G_i-G'_j)² + (B_i-B'_j)²
       + λ[(r_i-r_j)² + (c_i-c_j)²].
```

### Decision variables

```text
x_ij = 1 if source pixel i is placed at target position j; 0 otherwise.
```

There are `n²=40,960,000` decision variables at the default resolution.

### Objective function

```text
Minimize Z = Σ(i∈P) Σ(j∈P) C_ij x_ij.
```

The first part rewards matching the target's red, green, and blue values. The
second part penalizes moving a source pixel far from its original position.

### Constraints

Use every source pixel exactly once:

```text
Σ(j∈P) x_ij = 1             for all i∈P.
```

Fill every target position exactly once:

```text
Σ(i∈P) x_ij = 1             for all j∈P.
```

Bounds:

```text
0 ≤ x_ij ≤ 1                for all i,j∈P.
```

This is a balanced linear assignment problem, a special transportation LP.
Its constraint matrix is totally unimodular, so the LP relaxation has a binary
optimal solution without explicitly imposing integer restrictions.

## Method

1. Crop the source and target to squares and resize both to 80×80.
2. Calculate an RGB mismatch and spatial movement cost for every source-target
   pixel pair.
3. Solve the minimum-cost one-to-one assignment using SciPy.
4. Place each intact source RGB value at its assigned target position.
5. Animate source pixels moving from their original coordinates to their assigned
   coordinates.

The constant `PROXIMITY` controls the artistic tradeoff. Lower values favor
the target portrait; higher values retain more of the source's spatial layout.

## Why this is closer to Obamify

The open-source Obamify documentation allows both source and target images. Its
published optimal algorithm uses a one-to-one pixel assignment with a cost that
combines RGB mismatch and spatial proximity. Therefore independent red, green,
and blue mass transport was not the right model for reproducing this effect.

This implementation follows the same mathematical category and visual idea in
Python. It is a compact educational reimplementation rather than a line-by-line
port of the Rust application.
