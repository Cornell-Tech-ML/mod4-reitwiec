from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, Dict


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    x_f: Any = [v for v in vals]
    x_f[arg] += epsilon

    x_b: Any = [v for v in vals]
    x_b[arg] -= epsilon

    return (f(*x_f) - f(*x_b)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:  # pyright: ignore
        """Accumulates the given derivative `x` into the current node's total derivative."""
        pass

    @property
    def unique_id(self) -> int:  # pyright: ignore
        """Returns the unique identifier for this variable/node."""
        pass

    def is_leaf(self) -> bool:  # pyright: ignore
        """Determines if the current variable/node is a leaf in the computation graph."""
        pass

    def is_constant(self) -> bool:  # pyright: ignore
        """Checks if the variable is a constant."""
        pass

    @property
    def parents(self) -> Iterable["Variable"]:  # pyright: ignore
        """Returns the parent variables of this node in the computation graph."""
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:  # pyright: ignore
        """Computes the partial derivatives of input variables using the chain rule."""
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    traversed: Dict[int, Variable] = dict()

    def traverse(variable: Variable) -> None:
        if variable.unique_id in traversed.keys() or variable.is_constant():
            return
        if not variable.is_leaf():
            for parent in variable.parents:
                traverse(parent)

        traversed[variable.unique_id] = variable

    traverse(variable)
    return list(traversed.values())[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable : The right-most variable.
        deriv : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    sorted_variables: Iterable[Variable] = topological_sort(variable)
    gradient_map: Dict[int, Any] = {variable.unique_id: deriv}

    for current_variable in sorted_variables:
        if current_variable.is_leaf():
            current_variable.accumulate_derivative(
                gradient_map[current_variable.unique_id]
            )
        else:
            backward_gradients = current_variable.chain_rule(
                gradient_map[current_variable.unique_id]
            )
            for input_variable, partial_deriv in backward_gradients:
                if input_variable.unique_id in gradient_map:
                    gradient_map[input_variable.unique_id] += partial_deriv
                else:
                    gradient_map[input_variable.unique_id] = partial_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return saved tensors"""
        return self.saved_values
