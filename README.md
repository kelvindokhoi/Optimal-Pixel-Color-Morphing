# Optimal Pixel Morphing — Obamify-Style Assignment

This project rearranges the intact pixels of `MariaSiopsis.jpg` so they resemble
the structure of `obama_target.png`. It is the two-image behavior requested:

- **Source image:** supplies every output pixel color.
- **Target image:** determines where those colors should be placed.
- **Output image:** looks like the target but uses only the source's pixels.

The default target was copied from the open-source Obamify project's embedded
target image for this educational implementation. Obamify is MIT-licensed; see
<https://github.com/Spu7Nix/obamify>.

## Why this is a linear program

Let `i` index source pixels and `j` index target positions.

### Decision variables

```text
x[i,j] = 1 if source pixel i is assigned to target position j; otherwise 0.
```

On the default 80×80 grid, there are `6400² = 40,960,000` variables. This easily
satisfies an “at least three decision variables” requirement.

### Objective

```text
minimize  Σ(i) Σ(j) [color_cost(i,j) + λ spatial_cost(i,j)] x[i,j]
```

where:

```text
color_cost(i,j)   = (R_i-R'_j)² + (G_i-G'_j)² + (B_i-B'_j)²
spatial_cost(i,j) = (row_i-row_j)² + (column_i-column_j)²
```

The parameter `λ` is `--proximity`. A larger value keeps pixels closer to their
original positions, producing a subtler transformation. A smaller value makes
the rearranged image resemble the target more strongly.

### Constraints

Every source pixel is used exactly once:

```text
Σ(j) x[i,j] = 1       for every source pixel i
```

Every target position receives exactly one source pixel:

```text
Σ(i) x[i,j] = 1       for every target position j
```

Bounds:

```text
0 ≤ x[i,j] ≤ 1
```

Although `x[i,j]` is described as binary, the assignment constraint matrix is
totally unimodular. Therefore the ordinary LP relaxation automatically has an
integral optimal solution. SciPy's linear-assignment solver exploits this
special structure much more efficiently than a generic LP solver.

## Run

```powershell
cd "C:\Users\Kelvin\Desktop\Folders\Fall_2026_Books\Optimal_Pixel_Color_Morphing"
python optimal_rgb_transport.py
```

To morph different images, change the constants near the top of
`optimal_rgb_transport.py`:

```python
SOURCE_PATH = Path("source.jpg")
TARGET_PATH = Path("target.jpg")
```

To control the target resemblance/movement tradeoff:

```python
PROXIMITY = 0.01  # stronger target resemblance
PROXIMITY = 0.10  # preserve more source layout
```

## Faster multiscale version

`optimal_rgb_transport_multiscale.py` first solves a small exact assignment at
`COARSE_GRID`, then keeps only a local neighborhood of candidate source pixels
for each fine target position. It uses sparse bipartite matching on that pruned
graph. This is much more scalable than the full dense assignment, but the fine
solution is guaranteed optimal only among the retained candidate edges.

```powershell
python optimal_rgb_transport_multiscale.py
```

Adjust these settings inside that file:

```python
COARSE_GRID = 10
LOCAL_RADIUS = 6
```

Increasing `LOCAL_RADIUS` improves the chance of finding the unrestricted
solution but retains more edges and takes longer.

## Outputs

- `outputs/pixel_morph.gif` — animated source-to-target rearrangement
- `outputs/morphed_result.png` — final image
- `outputs/source_target_result.png` — source, target, and result comparison

## Relationship to Method 1

The RGB channels appear separately inside the color-matching coefficient, but
whole pixels are transported together. Moving R, G, and B independently creates
a channel-splitting effect; it does **not** preserve the source image's palette
and is not the Obamify-style behavior. The corrected assignment model is the
appropriate formulation for morphing one complete image into another.
