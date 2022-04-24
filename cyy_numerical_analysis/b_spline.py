import functools
import math

from cyy_numerical_analysis.polynomial import Polynomial


def limit(expr) -> float:
    match expr:
        case(p, q):
            p_coefficients = p.coefficients
            q_coefficients = q.coefficients
            while (
                p_coefficients
                and p_coefficients[0] == 0
                and q_coefficients
                and q_coefficients[0] == 0
            ):
                p_coefficients = p_coefficients[1:]
                q_coefficients = q_coefficients[1:]
            assert p_coefficients and q_coefficients
            if q_coefficients[0] != 0:
                return p_coefficients[0] / q_coefficients[0]
            # deal with numerical errors
            while (
                p_coefficients
                and abs(p_coefficients[0]) < 1e-20
                and q_coefficients
                and q_coefficients[0] == 0
            ):
                p_coefficients = p_coefficients[1:]
                q_coefficients = q_coefficients[1:]
            assert p_coefficients and q_coefficients
            if q_coefficients[0] != 0:
                return p_coefficients[0] / q_coefficients[0]
            raise RuntimeError(f"failed to take limit of {expr}")
    raise RuntimeError(f"failed to take limit of {expr}")


def simplify(expr) -> tuple[Polynomial, Polynomial]:
    """simplify expression"""
    match expr:
        case int() | float():
            return (Polynomial([expr]), Polynomial([1]))
        case Polynomial():
            return (expr, Polynomial([1]))
        case(op, a, b):
            expr = (op, simplify(a), simplify(b))
    match expr:
        case(op, (p1, q1), (p2, q2)):
            match op:
                case "+":
                    expr = (p1 * q2 + p2 * q1, q1 * q2)
                case "-":
                    expr = (p1 * q2 - p2 * q1, q1 * q2)
                case "*":
                    expr = (p1 * p2, q1 * q2)
                case "/":
                    expr = (p1 * q2, q1 * p2)
    match expr:
        case(p, q):
            p_coefficients = p.coefficients
            q_coefficients = q.coefficients
            while (
                p_coefficients
                and p_coefficients[0] == 0
                and q_coefficients
                and q_coefficients[0] == 0
            ):
                p_coefficients = p_coefficients[1:]
                q_coefficients = q_coefficients[1:]
            if not p_coefficients:
                p_coefficients = [0]
            expr = (Polynomial(p_coefficients), Polynomial(q_coefficients))
    return expr


class KnotVector:
    def __init__(self, knot_vector: list, degree):
        # if there are multiple points, add an epsilon to them, but we start from knot_vector[degree], because if knot_vector are all zeros and we start from knot_vector[0], then knot_vector[degree] would be greater than zero, and t=0 would be out of range.
        epsilon_cnts = [0] * len(knot_vector)
        for i in range(degree, len(knot_vector) - 1):
            assert knot_vector[i] <= knot_vector[i + 1]
            if knot_vector[i] == knot_vector[i + 1]:
                epsilon_cnts[i + 1] = epsilon_cnts[i] + 1
        for i in range(degree - 1):
            assert knot_vector[i] <= knot_vector[i + 1]
            if knot_vector[i] == knot_vector[i + 1]:
                epsilon_cnts[i + 1] = epsilon_cnts[i] + 1
        for i in reversed(range(degree)):
            if knot_vector[i] == knot_vector[i + 1]:
                epsilon_cnts[i] = epsilon_cnts[i + 1] - 1

        self.__knot_vector = knot_vector
        self.__epsilon_cnts = epsilon_cnts
        self.__degree = degree

    def __len__(self):
        return len(self.__knot_vector)

    def get_raw_kot_vector(self):
        return self.__knot_vector

    def get_knot(self, idx) -> Polynomial:
        return Polynomial(
            coefficients=(self.__knot_vector[idx], self.__epsilon_cnts[idx])
        )

    def get_knot_coefficients(self, idx) -> tuple:
        return (self.get_knot(idx).coefficients + (0,))[:2]

    def in_interval(self, t, index) -> bool:
        left_endpoint = self.get_knot_coefficients(index)
        parameter = None
        match t:
            case Polynomial():
                parameter = (t.coefficients + (0,))[:2]
            case _:
                parameter = (t, 0)

        right_endpoint = self.get_knot_coefficients(index + 1)
        return left_endpoint <= parameter < right_endpoint

    def get_parameter_index(self, t, point_num):
        for i in range(self.__degree, point_num + 2):
            if self.in_interval(t, index=i):
                return i
        raise RuntimeError(f"argument {t} out of range")

    def evaluate_base_function(self, t: int | float, index, degree):
        return self.evaluate_base_function_derivative(
            t, index, degree, derivative_degree=0
        )

    def evaluate_base_function_derivative(
        self, t: int | float, index, degree, derivative_degree
    ):
        return limit(
            self.__evaluate_base_function_derivative(
                t, index, degree, derivative_degree
            )
        )

    @functools.lru_cache(1000)
    def __evaluate_base_function_derivative(
        self, t: int | float, index, degree, derivative_degree
    ):
        assert derivative_degree >= 0
        if degree == 0:
            if derivative_degree > 0:
                return 0
            if self.in_interval(t, index):
                return 1
            return 0
        result = None
        for part_derivative_degree in range(derivative_degree + 1):
            if derivative_degree - part_derivative_degree > 1:
                continue
            tmp = simplify(
                (
                    "*",
                    math.comb(derivative_degree, part_derivative_degree),
                    self.__evaluate_base_function_derivative(
                        t, index, degree - 1, derivative_degree=part_derivative_degree
                    ),
                )
            )
            if derivative_degree - part_derivative_degree == 0:
                tmp = simplify(
                    (
                        "*",
                        tmp,
                        (
                            "/",
                            ("-", t, self.get_knot(index)),
                            ("-", self.get_knot(index + degree), self.get_knot(index)),
                        ),
                    )
                )
            else:
                assert derivative_degree - part_derivative_degree == 1
                tmp = simplify(
                    (
                        "*",
                        tmp,
                        (
                            "/",
                            1,
                            ("-", self.get_knot(index + degree), self.get_knot(index)),
                        ),
                    )
                )
            if result is None:
                result = tmp
            else:
                result = simplify(("+", result, tmp))
        for part_derivative_degree in range(derivative_degree + 1):
            if derivative_degree - part_derivative_degree > 1:
                continue
            tmp = simplify(
                (
                    "*",
                    math.comb(derivative_degree, part_derivative_degree),
                    self.__evaluate_base_function_derivative(
                        t,
                        index + 1,
                        degree - 1,
                        derivative_degree=part_derivative_degree,
                    ),
                )
            )
            if derivative_degree - part_derivative_degree == 0:
                tmp = simplify(
                    (
                        "*",
                        tmp,
                        (
                            "/",
                            ("-", self.get_knot(index + degree + 1), t),
                            (
                                "-",
                                self.get_knot(index + degree + 1),
                                self.get_knot(index + 1),
                            ),
                        ),
                    )
                )
            else:
                assert derivative_degree - part_derivative_degree == 1
                tmp = simplify(
                    (
                        "*",
                        tmp,
                        (
                            "/",
                            -1,
                            (
                                "-",
                                self.get_knot(index + degree + 1),
                                self.get_knot(index + 1),
                            ),
                        ),
                    )
                )
            result = simplify(("+", result, tmp))
        return result


class BSpline:
    def __init__(self, points: list, degree: int, knot_vector: list):
        assert len(knot_vector) == len(points) + degree + 1
        point_dimension = 1
        match points[0]:
            case[*_]:
                point_dimension = len(points[0])

        self.__point_dimension = point_dimension
        if point_dimension == 1:
            points = tuple((p,) for p in points)
        self.__points = points
        assert degree >= 1
        self.__knot_vector = KnotVector(knot_vector, degree=degree)
        self.__degree = degree

    @property
    def points(self):
        return self.__points

    @property
    def knot_vector(self):
        return self.__knot_vector.get_raw_kot_vector()

    def __call__(self, t: float):
        res = [
            limit(self.__evaluate(t, degree=self.__degree, point_index=i))
            for i in range(self.__point_dimension)
        ]
        if len(res) == 1:
            return res[0]
        return res

    def get_knot(self, index):
        return self.__knot_vector.get_knot(index)

    def get_knot_coefficients(self, index):
        return self.__knot_vector.get_knot_coefficients(index)

    @functools.lru_cache(1000)
    def __evaluate(self, t, degree, point_index, index=None):
        """implement de Casteljau algorithm"""
        if index is None:
            for i in range(degree, len(self.__knot_vector) - 1):
                left_endpoint = self.get_knot_coefficients(i)
                right_endpoint = self.get_knot_coefficients(i + 1)
                if left_endpoint <= (t, 0) <= right_endpoint:
                    index = i
                    break
        if index is None:
            raise RuntimeError(f"argument {t} out of range")

        if degree == 0:
            return self.__points[index][point_index]
        return simplify(
            (
                "+",
                (
                    "*",
                    (
                        "-",
                        1,
                        (
                            (
                                "/",
                                ("-", t, self.get_knot(index)),
                                (
                                    "-",
                                    self.get_knot(index + self.__degree + 1 - degree),
                                    self.get_knot(index),
                                ),
                            )
                        ),
                    ),
                    self.__evaluate(
                        t=t, degree=degree - 1, point_index=point_index, index=index - 1
                    ),
                ),
                (
                    "*",
                    (
                        "/",
                        ("-", t, self.get_knot(index)),
                        (
                            "-",
                            self.get_knot(index + self.__degree + 1 - degree),
                            self.get_knot(index),
                        ),
                    ),
                    self.__evaluate(
                        t=t, degree=degree - 1, point_index=point_index, index=index
                    ),
                ),
            )
        )
