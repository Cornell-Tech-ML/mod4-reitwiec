from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the scalar function to the given values.

        Args:
        ----
        cls : Class
            The class of the scalar function to be applied.
        vals : ScalarLike
            One or more scalar values to which the function will be applied.

        Returns:
        -------
        Scalar
            The result of applying the scalar function to the provided values.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the sum of two input values.

        Args:
        ----
        ctx : Context
            Context for saving values (not used here).
        a : float
            First input value.
        b : float
            Second input value.

        Returns:
        -------
        float
            Sum of `a` and `b`.

        """
        return operators.add(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the derivatives of the addition function.

        Args:
        ----
        ctx : Context
            Context with saved values (not used here).
        d_output : float
            Derivative of the output.

        Returns:
        -------
        Tuple[float, ...]
            Derivatives with respect to both inputs, equal to `d_output`.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the logarithm of the input value.

        Args:
        ----
        ctx : Context
            Context for saving values.
        a : float
            Input value.

        Returns:
        -------
        float
            Logarithm of `a`.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the derivative of the logarithm function.

        Args:
        ----
        ctx : Context
        Context with saved values.
        d_output : float
        Derivative of the output.

        Returns:
        -------
        float
        Derivative with respect to input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Mul function $f(x,y) = x * y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the product of two input values.

        Args:
        ----
        ctx : Context
            Context for saving values.
        a : float
            First input value.
        b : float
            Second input value.

        Returns:
        -------
        float
            Product of `a` and `b`.

        """
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the derivatives of the multiplication function.

        Args:
        ----
        ctx : Context
            Context with saved values.
        d_output : float
            Derivative of the output.

        Returns:
        -------
        Tuple[float, ...]
            Derivatives with respect to each input.

        """
        (a, b) = ctx.saved_values
        return operators.mul(b, d_output), operators.mul(a, d_output)


class Inv(ScalarFunction):
    """Inv function $f(x) = 1/x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the inverse of the input value.

        Args:
        ----
        ctx : Context
            Context for saving values.
        a : float
            Input value.

        Returns:
        -------
        float
            Inverse of `a`.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the derivative of the inverse function.

        Args:
        ----
        ctx : Context
            Context with saved values.
        d_output : float
            Derivative of the output.

        Returns:
        -------
        float
            Derivative with respect to input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Neg function $f(x) = -x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the negation of the input value.

        Args:
        ----
        ctx : Context
            Context for saving values (not used here).
        a : float
            Input value.

        Returns:
        -------
        float
            Negation of `a`.

        """
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the derivative of the negation function.

        Args:
        ----
        ctx : Context
            Context with saved values (not used here).
        d_output : float
            Derivative of the output.

        Returns:
        -------
        float
            Negated derivative.

        """
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the sigmoid of the input value.

        Args:
        ----
        ctx : Context
            Context for saving values.
        a : float
            Input value.

        Returns:
        -------
        float
            Sigmoid of `a`.

        """
        sig_a: float = operators.sigmoid(a)
        ctx.save_for_backward(sig_a)
        return sig_a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the derivative of the sigmoid function.

        Args:
        ----
        ctx : Context
            Context with saved values.
        d_output : float
            Derivative of the output.

        Returns:
        -------
        float
            Derivative with respect to input.

        """
        (sig_a,) = ctx.saved_values
        return operators.mul(operators.mul(sig_a, operators.add(1, (-sig_a))), d_output)


class ReLU(ScalarFunction):
    """ReLU function $f(x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the ReLU of the input value.

        Args:
        ----
        ctx : Context
            Context for saving values.
        a : float
            Input value.

        Returns:
        -------
        float
            ReLU of `a`.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the derivative of the ReLU function.

        Args:
        ----
        ctx : Context
            Context with saved values.
        d_output : float
            Derivative of the output.

        Returns:
        -------
        float
            Derivative with respect to input.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exp function $f(x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the exponential of the input value.

        Args:
        ----
        ctx : Context
            Context for saving values.
        a : float
            Input value.

        Returns:
        -------
        float
            Exponential of `a`.

        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the derivative of the exponential function.

        Args:
        ----
        ctx : Context
            Context with saved values.
        d_output : float
            Derivative of the output.

        Returns:
        -------
        float
            Derivative with respect to input.

        """
        (a,) = ctx.saved_values
        return operators.mul(operators.exp(a), d_output)


class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the less-than comparison between two values.

        Args:
        ----
        ctx : Context
            Context for saving values (not used here).
        a : float
            First input value.
        b : float
            Second input value.

        Returns:
        -------
        float
            1.0 if `a` is less than `b`, otherwise 0.0.

        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the derivative of the less-than function.

        Args:
        ----
        ctx : Context
            Context with saved values (not used here).
        d_output : float
            Derivative of the output (not used here).

        Returns:
        -------
        Tuple[float, float]
            Derivatives with respect to both inputs, always 0.0.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the equality comparison between two values.

        Args:
        ----
        ctx : Context
            Context for saving values (not used here).
        a : float
            First input value.
        b : float
            Second input value.

        Returns:
        -------
        float
            1.0 if `a` is equal to `b`, otherwise 0.0.

        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the derivative of the equality function.

        Args:
        ----
        ctx : Context
            Context with saved values (not used here).
        d_output : float
            Derivative of the output (not used here).

        Returns:
        -------
        Tuple[float, float]
            Derivatives with respect to both inputs, always 0.0.

        """
        return 0.0, 0.0
