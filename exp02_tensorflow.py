"""Experiment 2: Basic Python operations and TensorFlow 2 tensor operations."""
from __future__ import annotations
from typing import Any


def run_tensorflow_demo() -> dict[str, Any]:
    """Create tensors and demonstrate their properties and common operations."""
    import numpy as np
    import tensorflow as tf

    scalar = tf.constant(7, dtype=tf.int32)
    vector = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    matrix = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    tensor3d = tf.reshape(tf.range(1, 25, dtype=tf.float32), (2, 3, 4))
    a = tf.constant([10.0, 20.0, 30.0])
    b = tf.constant([2.0, 4.0, 5.0])

    def val(t):
        return t.numpy().tolist()

    tensors = {
        "scalar": {"value": val(scalar), "shape": scalar.shape.as_list(), "rank": int(tf.rank(scalar).numpy()), "dtype": scalar.dtype.name},
        "vector": {"value": val(vector), "shape": vector.shape.as_list(), "rank": int(tf.rank(vector).numpy()), "dtype": vector.dtype.name},
        "matrix": {"value": val(matrix), "shape": matrix.shape.as_list(), "rank": int(tf.rank(matrix).numpy()), "dtype": matrix.dtype.name},
        "tensor_3d": {"value": val(tensor3d), "shape": tensor3d.shape.as_list(), "rank": int(tf.rank(tensor3d).numpy()), "dtype": tensor3d.dtype.name},
    }
    operations = {
        "a": val(a), "b": val(b),
        "addition": val(a + b),
        "subtraction": val(a - b),
        "multiplication": val(a * b),
        "division": val(a / b),
        "square_a": val(tf.square(a)),
        "sqrt_a": val(tf.sqrt(a)),
        "mean_a": float(tf.reduce_mean(a).numpy()),
        "maximum_a": float(tf.reduce_max(a).numpy()),
        "reshape_vector_to_3x1": val(tf.reshape(vector, (3, 1))),
        "numpy_conversion_type": str(type(vector.numpy()).__name__),
    }
    # Include a basic pure-Python demonstration too.
    python_demo = {"integer_division": 17 // 5, "remainder": 17 % 5, "list_comprehension_squares": [n * n for n in range(1, 6)]}
    return {"tensorflow_version": tf.__version__, "numpy_version": np.__version__, "tensors": tensors, "operations": operations, "python_demo": python_demo}


if __name__ == "__main__":
    import pprint
    pprint.pp(run_tensorflow_demo())
