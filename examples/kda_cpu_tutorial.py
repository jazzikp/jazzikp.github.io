#!/usr/bin/env python3
"""Dependency-free numerical examples for Kimi Delta Attention (KDA).

Run from the site repository root with:
    python3 examples/kda_cpu_tutorial.py
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Vector = list[float]
Matrix = list[list[float]]


def vector_add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x + y for x, y in zip(a, b)]


def vector_sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x - y for x, y in zip(a, b)]


def vector_scale(scale: float, x: Sequence[float]) -> Vector:
    return [scale * value for value in x]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def outer(a: Sequence[float], b: Sequence[float]) -> Matrix:
    return [[x * y for y in b] for x in a]


def zeros(rows: int, columns: int) -> Matrix:
    return [[0.0 for _ in range(columns)] for _ in range(rows)]


def identity(size: int) -> Matrix:
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def matrix_add(a: Matrix, b: Matrix) -> Matrix:
    return [vector_add(a_row, b_row) for a_row, b_row in zip(a, b)]


def matrix_sub(a: Matrix, b: Matrix) -> Matrix:
    return [vector_sub(a_row, b_row) for a_row, b_row in zip(a, b)]


def matrix_scale(scale: float, matrix: Matrix) -> Matrix:
    return [vector_scale(scale, row) for row in matrix]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    columns = list(zip(*b))
    return [[dot(row, column) for column in columns] for row in a]


def matrix_vector(matrix: Matrix, vector: Sequence[float]) -> Vector:
    return [dot(row, vector) for row in matrix]


def transpose_matrix_vector(matrix: Matrix, vector: Sequence[float]) -> Vector:
    return [dot(column, vector) for column in zip(*matrix)]


def row_scale(scales: Sequence[float], matrix: Matrix) -> Matrix:
    return [vector_scale(scale, row) for scale, row in zip(scales, matrix)]


def max_abs_difference(a: Matrix, b: Matrix) -> float:
    return max(abs(x - y) for a_row, b_row in zip(a, b) for x, y in zip(a_row, b_row))


def format_vector(vector: Sequence[float], digits: int = 4) -> str:
    return "[" + ", ".join(f"{value:.{digits}f}" for value in vector) + "]"


def format_matrix(matrix: Matrix, digits: int = 4) -> str:
    return "[" + ",\n ".join(format_vector(row, digits) for row in matrix) + "]"


def l2_normalize(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(dot(vector, vector))
    return [value / norm for value in vector]


def kda_step(
    state: Matrix,
    query: Sequence[float],
    key: Sequence[float],
    value: Sequence[float],
    retention: Sequence[float],
    beta: float,
) -> tuple[Matrix, Vector, dict[str, Matrix | Vector]]:
    """Apply Eq. 1 as decay followed by a delta-rule correction."""
    decayed_state = row_scale(retention, state)
    prediction = transpose_matrix_vector(decayed_state, key)
    error = vector_sub(value, prediction)
    correction = matrix_scale(beta, outer(key, error))
    new_state = matrix_add(decayed_state, correction)
    output = transpose_matrix_vector(new_state, query)
    trace: dict[str, Matrix | Vector] = {
        "decayed_state": decayed_state,
        "prediction": prediction,
        "error": error,
        "correction": correction,
    }
    return new_state, output, trace


def vanilla_linear_step(state: Matrix, key: Sequence[float], value: Sequence[float]) -> Matrix:
    return matrix_add(state, outer(key, value))


def recurrent_kda(
    queries: Matrix,
    keys: Matrix,
    values: Matrix,
    log_retentions: Matrix,
    betas: Sequence[float],
    initial_state: Matrix,
) -> tuple[Matrix, Matrix]:
    state = [row[:] for row in initial_state]
    outputs: Matrix = []
    for query, key, value, log_retention, beta in zip(
        queries, keys, values, log_retentions, betas
    ):
        retention = [math.exp(g) for g in log_retention]
        state, output, _ = kda_step(state, query, key, value, retention, beta)
        outputs.append(output)
    return outputs, state


def official_naive_chunk_kda(
    queries: Matrix,
    keys: Matrix,
    values: Matrix,
    log_retentions: Matrix,
    betas: Sequence[float],
    initial_state: Matrix,
) -> tuple[Matrix, Matrix]:
    """Single-head, single-chunk port of FLA's naive_chunk_kda reference.

    It implements the UT-transformed, chunk-parallel equations with ordinary
    Python lists. The calculation is intentionally explicit rather than fast.
    """
    chunk_size = len(queries)
    key_size = len(keys[0])
    value_size = len(values[0])

    cumulative_g: Matrix = []
    running_g = [0.0] * key_size
    for step_g in log_retentions:
        running_g = vector_add(running_g, step_g)
        cumulative_g.append(running_g[:])

    # Causal triangular solve that produces the UT transform.
    transform = zeros(chunk_size, chunk_size)
    for row in range(chunk_size):
        for column in range(row):
            relative_decay = [
                math.exp(cumulative_g[row][channel] - cumulative_g[column][channel])
                for channel in range(key_size)
            ]
            similarity = dot(
                [keys[column][channel] * relative_decay[channel] for channel in range(key_size)],
                keys[row],
            )
            transform[row][column] = -betas[row] * similarity

    for row in range(1, chunk_size):
        old_row = transform[row][:]
        extra = [
            sum(old_row[middle] * transform[middle][column] for middle in range(chunk_size))
            for column in range(row)
        ]
        for column in range(row):
            transform[row][column] += extra[column]

    for row in range(chunk_size):
        transform[row][row] += 1.0
        for column in range(chunk_size):
            transform[row][column] *= betas[column]

    decayed_keys = [
        [math.exp(cumulative_g[row][channel]) * keys[row][channel] for channel in range(key_size)]
        for row in range(chunk_size)
    ]
    w = matmul(transform, decayed_keys)
    u = matmul(transform, values)
    pseudo_values = matrix_sub(u, matmul(w, initial_state))

    causal_scores = zeros(chunk_size, chunk_size)
    for query_position in range(chunk_size):
        for key_position in range(query_position + 1):
            relative_decay = [
                math.exp(
                    cumulative_g[query_position][channel]
                    - cumulative_g[key_position][channel]
                )
                for channel in range(key_size)
            ]
            decayed_query = [
                queries[query_position][channel] * relative_decay[channel]
                for channel in range(key_size)
            ]
            causal_scores[query_position][key_position] = dot(decayed_query, keys[key_position])

    inter_chunk_outputs = [
        transpose_matrix_vector(
            initial_state,
            [
                queries[position][channel] * math.exp(cumulative_g[position][channel])
                for channel in range(key_size)
            ],
        )
        for position in range(chunk_size)
    ]
    intra_chunk_outputs = matmul(causal_scores, pseudo_values)
    outputs = matrix_add(inter_chunk_outputs, intra_chunk_outputs)

    final_decay = [math.exp(value) for value in cumulative_g[-1]]
    final_state = row_scale(final_decay, initial_state)
    for position in range(chunk_size):
        decay_to_end = [
            math.exp(cumulative_g[-1][channel] - cumulative_g[position][channel])
            for channel in range(key_size)
        ]
        effective_key = [
            keys[position][channel] * decay_to_end[channel]
            for channel in range(key_size)
        ]
        final_state = matrix_add(final_state, outer(effective_key, pseudo_values[position]))

    return outputs, final_state


def token_transition(
    key: Sequence[float], retention: Sequence[float], beta: float
) -> Matrix:
    """Return M_t = (I - beta k k^T) Diag(alpha)."""
    size = len(key)
    return [
        [
            ((1.0 if row == column else 0.0) - beta * key[row] * key[column])
            * retention[column]
            for column in range(size)
        ]
        for row in range(size)
    ]


def segment_summary(
    keys: Matrix,
    values: Matrix,
    retentions: Matrix,
    betas: Sequence[float],
) -> tuple[Matrix, Matrix]:
    """Summarize a segment as S_out = M_segment S_in + B_segment."""
    key_size = len(keys[0])
    value_size = len(values[0])
    cumulative_transition = identity(key_size)
    zero_start_state = zeros(key_size, value_size)
    for key, value, retention, beta in zip(keys, values, retentions, betas):
        transition = token_transition(key, retention, beta)
        write = matrix_scale(beta, outer(key, value))
        cumulative_transition = matmul(transition, cumulative_transition)
        zero_start_state = matrix_add(matmul(transition, zero_start_state), write)
    return cumulative_transition, zero_start_state


def compose_segments(
    first: tuple[Matrix, Matrix], second: tuple[Matrix, Matrix]
) -> tuple[Matrix, Matrix]:
    """Return second(first(S)); composition is associative."""
    first_transition, first_write = first
    second_transition, second_write = second
    return (
        matmul(second_transition, first_transition),
        matrix_add(matmul(second_transition, first_write), second_write),
    )


def apply_segment(summary: tuple[Matrix, Matrix], state: Matrix) -> Matrix:
    transition, zero_start_state = summary
    return matrix_add(matmul(transition, state), zero_start_state)


def stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def k3_retention(logit: float, log_scale: float = 0.0, g_min: float = -5.0) -> float:
    log_decay = g_min * stable_sigmoid(math.exp(log_scale) * logit)
    return math.exp(log_decay)


def kimi_linear_retention(logit: float, log_scale: float = 0.0) -> float:
    log_decay = -math.exp(log_scale) * softplus(logit)
    return math.exp(log_decay)


def heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def example_delta_overwrite() -> None:
    heading("1. Delta rule: learn, avoid duplicate accumulation, then overwrite")
    key = [1.0, 0.0]
    query = key
    value = [2.0, -1.0]
    state = zeros(2, 2)

    state, output, first = kda_step(state, query, key, value, [1.0, 1.0], 1.0)
    print("After first write, S =")
    print(format_matrix(state))
    print("Read with q=k:", format_vector(output))

    state, output, repeated = kda_step(state, query, key, value, [1.0, 1.0], 1.0)
    print("Repeated identical pair has prediction", format_vector(repeated["prediction"]))
    print("and residual error", format_vector(repeated["error"]), "so S is unchanged.")

    new_value = [7.0, 3.0]
    state, output, overwritten = kda_step(state, query, key, new_value, [1.0, 1.0], 1.0)
    print("New value for the same key gives residual", format_vector(overwritten["error"]))
    print("and the read is overwritten to", format_vector(output))

    linear_state = zeros(2, 2)
    linear_state = vanilla_linear_step(linear_state, key, value)
    linear_state = vanilla_linear_step(linear_state, key, value)
    print("Vanilla linear attention stores the duplicate twice and reads",
          format_vector(transpose_matrix_vector(linear_state, query)))


def example_beta_and_channel_decay() -> None:
    heading("2. Beta is a learning rate; alpha is a per-key-channel retention gate")
    state = [[2.0], [8.0]]
    key = [1.0, 0.0]
    state, output, trace = kda_step(
        state=state,
        query=key,
        key=key,
        value=[10.0],
        retention=[1.0, 1.0],
        beta=0.25,
    )
    print("Old prediction 2, target 10, beta 0.25 -> new prediction", format_vector(output))
    print("This is (1-beta)*2 + beta*10 = 4 for a unit-normalized key.")

    original_state = [[10.0, 1.0], [2.0, 20.0]]
    _, _, trace = kda_step(
        state=original_state,
        query=[0.0, 1.0],
        key=[0.0, 1.0],
        value=[2.0, 20.0],
        retention=[0.10, 0.90],
        beta=0.50,
    )
    print("\nBefore the delta correction, Diag(alpha) scales state rows:")
    print(format_matrix(original_state), "\nbecomes\n", format_matrix(trace["decayed_state"]))
    print("The first learned key-feature channel forgets 90%; the second forgets 10%.")
    print("Prediction after decay:", format_vector(trace["prediction"]))
    print("Residual written by the delta rule:", format_vector(trace["error"]))


def example_chunk_equivalence() -> None:
    heading("3. Chunk-parallel UT form equals token-by-token recurrence")
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    queries = [
        [1.0, 0.0],
        [0.0, 1.0],
        [inv_sqrt_2, inv_sqrt_2],
        [0.6, 0.8],
    ]
    keys = [
        [1.0, 0.0],
        [0.0, 1.0],
        [inv_sqrt_2, inv_sqrt_2],
        [0.6, 0.8],
    ]
    values = [[2.0, -1.0], [0.5, 3.0], [4.0, 1.0], [-2.0, 2.5]]
    retentions = [[0.95, 0.80], [0.70, 0.98], [0.90, 0.85], [0.99, 0.75]]
    log_retentions = [[math.log(value) for value in row] for row in retentions]
    betas = [0.7, 0.4, 0.8, 0.6]
    initial_state = [[0.3, -0.2], [0.1, 0.4]]

    recurrent_outputs, recurrent_state = recurrent_kda(
        queries, keys, values, log_retentions, betas, initial_state
    )
    chunk_outputs, chunk_state = official_naive_chunk_kda(
        queries, keys, values, log_retentions, betas, initial_state
    )
    output_error = max_abs_difference(recurrent_outputs, chunk_outputs)
    state_error = max_abs_difference(recurrent_state, chunk_state)

    print("Token-by-token outputs:\n", format_matrix(recurrent_outputs, 6))
    print("Chunk-parallel outputs:\n", format_matrix(chunk_outputs, 6))
    print(f"Maximum output difference: {output_error:.3e}")
    print(f"Maximum final-state difference: {state_error:.3e}")
    assert output_error < 1e-12
    assert state_error < 1e-12


def example_segment_composition() -> None:
    heading("4. Context parallelism: summarize and associatively compose segments")
    keys = [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [0.6, -0.8]]
    keys = [l2_normalize(key) for key in keys]
    values = [[1.0, 2.0], [3.0, -1.0], [0.5, 4.0], [-2.0, 1.5]]
    retentions = [[0.9, 0.8], [0.7, 0.95], [0.85, 0.9], [0.99, 0.75]]
    betas = [0.5, 0.8, 0.6, 0.7]
    initial_state = [[0.2, -0.1], [0.4, 0.3]]

    all_at_once = segment_summary(keys, values, retentions, betas)
    left = segment_summary(keys[:2], values[:2], retentions[:2], betas[:2])
    right = segment_summary(keys[2:], values[2:], retentions[2:], betas[2:])
    composed = compose_segments(left, right)

    direct_state = apply_segment(all_at_once, initial_state)
    composed_state = apply_segment(composed, initial_state)
    difference = max_abs_difference(direct_state, composed_state)
    print("Direct final state:\n", format_matrix(direct_state, 6))
    print("Composed two-segment state:\n", format_matrix(composed_state, 6))
    print(f"Maximum difference: {difference:.3e}")
    print("Each segment communicates (M_segment, S_generated_from_zero), not all tokens.")
    assert difference < 1e-12


def example_lower_bounded_decay() -> None:
    heading("5. Kimi K3's lower-bounded log-decay")
    print("At A=0 and g_min=-5:")
    print(" logit | old Kimi Linear alpha | Kimi K3 alpha")
    for logit in [-10.0, -2.0, 0.0, 2.0, 10.0, 100.0]:
        old_alpha = kimi_linear_retention(logit)
        new_alpha = k3_retention(logit)
        print(f"{logit:6.1f} | {old_alpha:21.6e} | {new_alpha:14.6e}")

    lower_bound = math.exp(-5.0)
    largest_tile_rescale = math.exp(16.0 * 5.0)
    bf16_max = (2.0 - 2.0 ** -7) * 2.0 ** 127
    print(f"\nPer-step K3 retention is greater than exp(-5) = {lower_bound:.6e}.")
    print(f"Worst 16-token reciprocal rescaling is below exp(80) = {largest_tile_rescale:.6e}.")
    print(f"Maximum finite BF16 value is approximately {bf16_max:.6e}.")
    print("The bound protects a 16-token compute tile; it does not stop decay over a long sequence.")
    assert largest_tile_rescale < bf16_max


def main() -> None:
    print("Kimi Delta Attention: executable numerical tutorial")
    example_delta_overwrite()
    example_beta_and_channel_decay()
    example_chunk_equivalence()
    example_segment_composition()
    example_lower_bounded_decay()
    print("\nAll numerical equivalence checks passed.")


if __name__ == "__main__":
    main()
