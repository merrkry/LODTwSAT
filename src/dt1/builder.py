import typing
import numpy
import sklearn.tree

from dt1.tree import DecisionTree
from dt1.types import FeatureMatrix, LabelVector

from pysat.formula import CNF, And, Equals, IDPool, Implies, Neg, Or
from pysat.card import CardEnc
from pysat.solvers import Solver


# Ideally this can be splitted into smaller functions for maintainability,
# but such abstraction would drastically increase the length of code.
# Considering this is only a reproduction, we focus on exposing the main idea.
def build_dt1_classifier(
    features: FeatureMatrix, labels: LabelVector, max_size: int | None
) -> DecisionTree | None:
    """
    Build a DT1 classifier using SAT encoding.
    :param features: training features
    :param labels: training labels
    :param max_size: maximum tree size, or None to auto-compute via CART
    :return: DecisionTree if found, None if max_size is user-provided and too strict
    """
    # Track if upper bound can be trusted (auto-computed = trusted)
    trusted_bound: bool = max_size is None

    if max_size is None:
        cart_tree = sklearn.tree.DecisionTreeClassifier().fit(features, labels)
        max_size = cart_tree.tree_.node_count
        trusted_bound = True

    max_size = typing.cast(int, max_size)

    # We assume root is not a leaf
    if max_size < 3:
        if trusted_bound:
            assert False, f"Auto-computed max_size={max_size} is invalid (must be >= 3)"
        return None

    n_samples = features.shape[0]
    assert n_samples == labels.shape[0]
    n_features = features.shape[1]

    last_tree: DecisionTree | None = None

    for size in range(3, max_size + 1, -1):
        if size % 2 == 0:
            continue

        vpool = IDPool()

        def var(*args) -> int:
            return vpool.id(args)

        def gen_LR(i: int) -> typing.Iterator[int]:
            """
            Set of possible left child node.
            """
            for j in range(i + 1, min(2 * i, size - 1) + 1):
                if j % 2 == 0:
                    yield j

        def gen_RR(i: int) -> typing.Iterator[int]:
            """
            Set of possible right child node.
            """
            for j in range(i + 2, min(2 * i + 1, size) + 1):
                if j % 2 == 1:
                    yield j

        def gen_P(i: int) -> typing.Iterator[int]:
            """
            Set of possible parent node.
            Note that the notation of `P` isn't directly used in the paper.
            """
            for j in range(i // 2, i):
                yield j

        # TODO: consider trying WCNF since we have cardinality constraints.
        encodings = CNF()

        # Below we use the same numbering of constraints as in the paper without further explanation.
        # See Section 3.1 and 3.2 for details.

        # (1)
        encodings.extend(Neg(var("v", 1)))

        # (2)
        encodings.extend(
            [
                Implies(var("v", i), Neg(var("l", i, j)))
                for i in range(1, size + 1)
                for j in gen_LR(i)
            ]
        )

        # (3)
        encodings.extend(
            [
                Equals(var("l", i, j), var("r", i, j + 1))
                for i in range(1, size + 1)
                for j in gen_LR(i)
            ]
        )

        # (4)
        def constraint_4():
            def make_sum(i: int):
                return CardEnc.equals(
                    [var("l", i, j) for j in gen_LR(i)],
                    bound=1,
                    vpool=vpool,
                )

            return [
                Implies(
                    Neg(var("v", i)),
                    make_sum(i),
                )
                for i in range(1, size + 1)
            ]

        encodings.extend(constraint_4())

        # (5)
        encodings.extend(
            [
                Equals(Neg(var("p", j, i)), var("l", i, j))
                for i in range(1, size + 1)
                for j in gen_LR(i)
            ]
        )
        encodings.extend(
            [
                Equals(Neg(var("p", j, i)), var("r", i, j + 1))
                for i in range(1, size + 1)
                for j in gen_RR(i)
            ]
        )

        # (6)
        encodings.extend(
            [
                CardEnc.equals(
                    [var("p", j, i) for i in gen_P(j)],
                    bound=1,
                    vpool=vpool,
                )
                for j in range(2, size + 1)
            ]
        )

        # (7) (8)
        for q in range(n_samples):
            if labels[q] == 0:
                encodings.extend(Neg(var("d", q, 1, 0)))
                encodings.extend(
                    [
                        Equals(
                            var("d", q, j, 0),
                            Or(
                                [
                                    inner
                                    for i in gen_P(j)
                                    for inner in (
                                        And(var("p", j, i), var("d", q, i, 0)),
                                        And(var("a", q, i), var("r", i, j)),
                                    )
                                ]
                            ),
                        )
                        for j in range(2, size + 1)
                    ]
                )
            else:
                encodings.extend(Neg(var("d", q, 1, 1)))
                encodings.extend(
                    [
                        Equals(
                            var("d", q, j, 1),
                            Or(
                                [
                                    inner
                                    for i in gen_P(j)
                                    for inner in (
                                        And(var("p", j, i), var("d", q, i, 1)),
                                        And(Neg(var("a", q, i)), var("l", i, j)),
                                    )
                                ]
                            ),
                        )
                        for j in range(2, size + 1)
                    ]
                )

        # (9)
        encodings.extend(
            [
                Implies(And(var("u", r, i), var("p", j, i)), Neg(var("a", r, j)))
                for j in range(1, size + 1)
                for r in range(n_samples)
                for i in gen_P(j)
            ]
        )
        encodings.extend(
            [
                Equals(
                    var("u", r, j),
                    Or(
                        var("a", r, j),
                        Or([And(var("u", r, i), var("p", j, i)) for i in gen_P(j)]),
                    ),
                )
                for j in range(1, size + 1)
                for r in range(n_samples)
            ]
        )

        # (10)
        encodings.extend(
            [
                Implies(
                    Neg(var("v", j)),
                    CardEnc.equals(
                        [var("a", r, j) for r in range(n_samples)],
                        bound=1,
                        vpool=vpool,
                    ),
                )
                for j in range(1, size + 1)
            ]
        )

        # (11)
        encodings.extend(
            [
                Implies(
                    var("v", j),
                    CardEnc.equals(
                        [var("a", r, j) for r in range(n_samples)],
                        bound=0,
                        vpool=vpool,
                    ),
                )
                for j in range(1, size + 1)
            ]
        )

        # (12) (13)
        for q in range(n_samples):
            if labels[q] == 1:
                encodings.extend(
                    [
                        Implies(
                            And(var("v", j), Neg(var("c", j))),
                            Or(
                                [
                                    var("d", r, j, features[q][r])
                                    for r in range(n_samples)
                                ]
                            ),
                        )
                        for j in range(1, size + 1)
                    ]
                )
            else:
                encodings.extend(
                    [
                        Implies(
                            And(var("v", j), var("c", j)),
                            Or(
                                [
                                    var("d", r, j, features[q][r])
                                    for r in range(n_samples)
                                ]
                            ),
                        )
                        for j in range(1, size + 1)
                    ]
                )

        # TODO: additional inference constraints in Section 3.3

        # TODO: replace with more modern solvers, especially those with native WCNF support
        with Solver(name="glucose3", bootstrap_with=encodings.clauses) as solver:
            unsat = not solver.solve()

            if unsat:
                break
            else:
                model = solver.get_model()

                left = numpy.array([0] * (size + 1), dtype=numpy.int32)
                right = numpy.array([0] * (size + 1), dtype=numpy.int32)
                node_feature = numpy.array([0] * (size + 1), dtype=numpy.int32)
                node_label = numpy.array([-1] * (size + 1), dtype=numpy.int32)
                model = typing.cast(list[typing.Any], model)
                for lit in model:
                    assignment: bool = lit > 0
                    var_id = abs(lit)
                    var_args = vpool.obj(var_id)
                    var_args = typing.cast(tuple[str, ...], var_args)
                    if var_args[0] == "l" and assignment:
                        i = int(var_args[1])
                        j = int(var_args[2])
                        left[i] = j
                    elif var_args[0] == "r" and assignment:
                        i = int(var_args[1])
                        j = int(var_args[2])
                        right[i] = j
                    elif var_args[0] == "a" and assignment:
                        r = int(var_args[1])
                        j = int(var_args[2])
                        node_feature[j] = r + 1  # 1-indexed
                    elif var_args[0] == "c" and assignment:
                        j = int(var_args[1])
                        node_label[j] = 1
                    else:
                        pass

                last_tree = DecisionTree(
                    left=left, right=right, features=node_feature, labels=node_label
                )

    if last_tree is None and trusted_bound:
        # Auto-computed upper bound should always yield a solution
        # Failure indicates a bug in the SAT encoding
        assert False, f"Failed to build tree with auto-computed max_size={max_size}"

    return last_tree
