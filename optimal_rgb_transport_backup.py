"""Obamify-style optimal pixel assignment in Python.

The source pixels keep their complete RGB colors. A linear assignment model
rearranges those pixels so their new positions resemble a target image while a
spatial penalty discourages unnecessarily long moves.

The mathematical decision variable is x[i,j]: 1 when source pixel i is placed
at target position j, and 0 otherwise. scipy.optimize.linear_sum_assignment
solves the equivalent linear-programming assignment problem efficiently.
"""

from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


# Project inputs and settings. Change these values directly when needed.
SOURCE_PATH = Path("Moe.jpg")
TARGET_PATH = Path("MariaSiopsis.jpg")
OUTPUT_DIR = Path("outputs")
GRID = 40
COARSE_GRID = 10
LOCAL_RADIUS = 6
PROXIMITY = 0.035
FRAMES = 31
DISPLAY_SIZE = 800
FRAME_DURATION_MS = 90
START_PAUSE_FRAMES = 6
END_PAUSE_FRAMES = 10
PAUSE_DURATION_MS = 120


def crop_square(image):
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def load_grid(path, grid):
    image = crop_square(Image.open(path).convert("RGB"))
    image = image.resize((grid, grid), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float64) / 255.0


def pixel_coordinates(grid):
    rows, cols = np.indices((grid, grid))
    denominator = max(grid - 1, 1)
    return np.column_stack((rows.ravel(), cols.ravel())).astype(float) / denominator


def build_cost_matrix(source,target,proximity,):
    grid = source.shape[0]
    source_colors = source.reshape(-1, 3).astype(np.float32)
    target_colors = target.reshape(-1, 3).astype(np.float32)
    coordinates = pixel_coordinates(grid).astype(np.float32)

    # ||target_j-source_i||² = ||target_j||² + ||source_i||² - 2 target_j·source_i.
    # Write the matrix product directly into the final array to avoid creating
    # another n-by-n temporary during the largest allocation.
    target_norms = np.einsum("ij,ij->i", target_colors, target_colors)
    source_norms = np.einsum("ij,ij->i", source_colors, source_colors)
    cost = np.empty((target_colors.shape[0], source_colors.shape[0]), dtype=np.float32)
    np.matmul(target_colors, source_colors.T, out=cost)
    cost *= -2.0
    cost += target_norms[:, None]
    cost += source_norms[None, :]

    coordinate_norms = np.einsum("ij,ij->i", coordinates, coordinates)
    spatial_cost = np.empty_like(cost)
    np.matmul(coordinates, coordinates.T, out=spatial_cost)
    spatial_cost *= -2.0
    spatial_cost += coordinate_norms[:, None]
    spatial_cost += coordinate_norms[None, :]
    cost += np.float32(proximity) * spatial_cost
    np.maximum(cost, 0.0, out=cost)
    return cost


def build_pruned_cost_matrix(source, target, coarse_assignment, proximity, local_radius):
    """Build a sparse fine-grid cost matrix around coarse predictions."""
    fine_grid = source.shape[0]
    coarse_grid = int(round(np.sqrt(coarse_assignment.size)))
    if fine_grid % coarse_grid != 0:
        raise ValueError("GRID must be divisible by COARSE_GRID")
    scale = fine_grid // coarse_grid
    pixel_count = fine_grid * fine_grid
    source_colors = source.reshape(-1, 3).astype(np.float32)
    target_colors = target.reshape(-1, 3).astype(np.float32)
    coordinates = pixel_coordinates(fine_grid).astype(np.float32)

    rows = []
    cols = []
    values = []
    for target_id in range(pixel_count):
        target_row, target_col = divmod(target_id, fine_grid)
        coarse_target_id = (target_row // scale) * coarse_grid + (target_col // scale)
        predicted_source_id = int(coarse_assignment[coarse_target_id])
        predicted_row, predicted_col = divmod(predicted_source_id, coarse_grid)
        center_row = predicted_row * scale + scale // 2
        center_col = predicted_col * scale + scale // 2

        row_min = max(0, center_row - local_radius)
        row_max = min(fine_grid - 1, center_row + local_radius)
        col_min = max(0, center_col - local_radius)
        col_max = min(fine_grid - 1, center_col + local_radius)
        candidate_ids = [
            row * fine_grid + col
            for row in range(row_min, row_max + 1)
            for col in range(col_min, col_max + 1)
        ]
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)

        color_delta = source_colors[candidate_ids] - target_colors[target_id]
        color_cost = np.sum(color_delta * color_delta, axis=1)
        spatial_delta = coordinates[candidate_ids] - coordinates[target_id]
        spatial_cost = np.sum(spatial_delta * spatial_delta, axis=1)
        candidate_costs = color_cost + np.float32(proximity) * spatial_cost

        rows.extend([target_id] * candidate_ids.size)
        cols.extend(candidate_ids.tolist())
        # Sparse matching treats explicit zeros as missing edges, so add a tiny
        # positive constant without changing meaningful cost comparisons.
        values.extend((candidate_costs + np.float32(1e-8)).tolist())

    return coo_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=(pixel_count, pixel_count),
    ).tocsr()


def solve_pruned_assignment(cost):
    """Solve the sparse one-to-one assignment on retained candidate edges."""
    target_ids, source_ids = min_weight_full_bipartite_matching(cost)
    if len(target_ids) != cost.shape[0]:
        raise RuntimeError(
            "Pruned candidate graph has no perfect matching; increase LOCAL_RADIUS."
        )
    assignment = np.empty(cost.shape[0], dtype=np.int64)
    assignment[target_ids] = source_ids
    return assignment, float(cost[target_ids, source_ids].sum())


def selected_cost_components(source,target,assignment):
    grid = source.shape[0]
    target_colors = target.reshape(-1, 3)
    assigned_source_colors = source.reshape(-1, 3)[assignment]
    color_cost = float(np.sum((target_colors - assigned_source_colors) ** 2))

    coordinates = pixel_coordinates(grid)
    assigned_source_coordinates = coordinates[assignment]
    spatial_cost = float(np.sum((coordinates - assigned_source_coordinates) ** 2))
    return color_cost, spatial_cost


def solve_assignment(cost):
    target_ids, source_ids = linear_sum_assignment(cost, maximize=False)
    assignment = np.empty(cost.shape[0], dtype=np.int64)
    assignment[target_ids] = source_ids
    return assignment, float(cost[target_ids, source_ids].sum())


def rearranged_image(source, assignment):
    source_colors = source.reshape(-1, 3)
    return source_colors[assignment].reshape(source.shape)


def to_uint8(array):
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def pixel_art(array, size):
    return Image.fromarray(to_uint8(array), mode="RGB").resize(
        (size, size), Image.Resampling.NEAREST
    )


def add_caption(image, text):
    caption_height = 44
    result = Image.new("RGB", (image.width, image.height + caption_height), "white")
    result.paste(image, (0, 0))
    ImageDraw.Draw(result).text((12, image.height + 13), text, fill="black")
    return result


def render_particle_frame(source,assignment,t,display_size,):
    grid = source.shape[0]
    pixel_count = grid * grid
    target_rows, target_cols = np.divmod(np.arange(pixel_count), grid)
    source_ids = assignment
    source_rows, source_cols = np.divmod(source_ids, grid)

    eased = t * t * (3.0 - 2.0 * t)
    rows = (1.0 - eased) * source_rows + eased * target_rows
    cols = (1.0 - eased) * source_cols + eased * target_cols
    colors = to_uint8(source.reshape(-1, 3)[source_ids])

    canvas = Image.new("RGB", (display_size, display_size), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    cell = display_size / grid
    particle_size = max(2, int(np.ceil(cell * 0.88)))

    # Draw longer-moving particles first; local pixels remain visually legible.
    movement = (target_rows - source_rows) ** 2 + (target_cols - source_cols) ** 2
    for index in np.argsort(-movement):
        center_x = (cols[index] + 0.5) * cell
        center_y = (rows[index] + 0.5) * cell
        half = particle_size / 2
        color = tuple(int(value) for value in colors[index])
        draw.rectangle(
            (
                int(center_x - half),
                int(center_y - half),
                int(center_x + half),
                int(center_y + half),
            ),
            fill=color,
        )
    return canvas


def save_comparison(source,target,result,output_path,display_size,source_name,target_name):
    panel_size = display_size // 2
    panels = [
        add_caption(pixel_art(source, panel_size), f"Source colors: {source_name}"),
        add_caption(pixel_art(target, panel_size), f"Target structure: {target_name}"),
        add_caption(pixel_art(result, panel_size), "Optimally rearranged source pixels"),
    ]
    comparison = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        comparison.paste(panel, (x, 0))
        x += panel.width
    comparison.save(output_path)


def main():
    if GRID < 2:raise ValueError("GRID must be at least 2")
    if PROXIMITY < 0:raise ValueError("PROXIMITY cannot be negative")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = load_grid(SOURCE_PATH, GRID)
    target = load_grid(TARGET_PATH, GRID)
    coarse_source = load_grid(SOURCE_PATH, COARSE_GRID)
    coarse_target = load_grid(TARGET_PATH, COARSE_GRID)
    pixel_count = GRID * GRID
    variable_count = pixel_count * pixel_count

    start = time.perf_counter()
    print(f"Solving coarse {COARSE_GRID}x{COARSE_GRID} exact assignment...")
    coarse_cost = build_cost_matrix(coarse_source, coarse_target, PROXIMITY)
    coarse_assignment, _ = solve_assignment(coarse_cost)

    print(
        f"Building pruned fine cost graph for {pixel_count:,} pixels "
        f"(full model would have {variable_count:,} x[i,j] variables)..."
    )
    pruned_cost = build_pruned_cost_matrix(
        source, target, coarse_assignment, PROXIMITY, LOCAL_RADIUS
    )
    print(f"Retained {pruned_cost.nnz:,} candidate edges.")
    assignment, objective = solve_pruned_assignment(pruned_cost)
    result = rearranged_image(source, assignment)

    selected_color_cost, selected_spatial_cost = selected_cost_components(source, target, assignment)

    comparison_path = OUTPUT_DIR / "source_target_result.png"
    save_comparison(
        source,
        target,
        result,
        comparison_path,
        DISPLAY_SIZE,
        SOURCE_PATH.stem,
        TARGET_PATH.stem,
    )
    pixel_art(result, DISPLAY_SIZE).save(
        OUTPUT_DIR / "morphed_result.png"
    )

    morph_frames = []
    for t in np.linspace(0.0, 1.0, FRAMES):
        frame = render_particle_frame(source, assignment, float(t), DISPLAY_SIZE)
        morph_frames.append(add_caption(frame, f"Optimal pixel assignment   t = {t:.2f}"))

    # Hold the source and final states briefly so both endpoints are easy to see.
    frames = (
        [morph_frames[0]] * START_PAUSE_FRAMES
        + morph_frames
        + [morph_frames[-1]] * END_PAUSE_FRAMES
    )
    durations = (
        [PAUSE_DURATION_MS] * START_PAUSE_FRAMES
        + [FRAME_DURATION_MS] * len(morph_frames)
        + [PAUSE_DURATION_MS] * END_PAUSE_FRAMES
    )
    gif_path = OUTPUT_DIR / "pixel_morph.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )

    elapsed = time.perf_counter() - start
    print(f"Combined objective: {objective:.6f}")
    print(f"  RGB mismatch:     {selected_color_cost:.6f}")
    print(f"  Movement total:   {selected_spatial_cost:.6f}")
    print(f"  Proximity weight: {PROXIMITY:.6f}")
    print(f"Finished in {elapsed:.2f} seconds.")
    print(f"Result:    {(OUTPUT_DIR / 'morphed_result.png').resolve()}")
    print(f"Animation: {gif_path.resolve()}")


if __name__ == "__main__":
    main()
