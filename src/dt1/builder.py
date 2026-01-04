import collections.abc
import typing

import nnf
import nnf.dimacs
from nnf.operators import iff, implies
from nnf.tseitin import to_CNF
import numpy
import sklearn.tree
from nnf import NNF, Aux, Var
from pysat.card import CardEnc
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

from dt1.tree import (
    NODE_LABEL_IRRELEVANT,
    NODE_LABEL_POSITIVE,
    NOTE_LABEL_NEGATIVE,
    DecisionTree,
)
from dt1.types import FeatureMatrix, LabelVector


# Ideally this can be splitted into smaller functions for maintainability,
# but such abstraction would drastically increase the length of code.
# Considering this is only a reproduction, we focus on exposing the main idea.
def _build_dt_from_fixed_size(
    features: FeatureMatrix,
    labels: LabelVector,
    size: int,
) -> DecisionTree | None:
    """
    Build a decision tree with exactly `size` nodes using SAT encoding.
    :param features: training features
    :param labels: training labels
    :param size: exact number of nodes in the tree (must be odd, >= 3)
    :param n_features: number of features
    :return: DecisionTree if SAT, None if UNSAT
    """
    import collections.abc

    n_samples = features.shape[0]
    n_features = features.shape[1]
    assert n_samples == labels.shape[0]

    vpool = IDPool()

    var_cache: dict[
        tuple[str, int] | tuple[str, int, int] | tuple[str, int, int, int] | tuple[Aux],
        Var,
    ] = {}

    def var(*args) -> Var:
        if args not in var_cache:
            var_cache[args] = Var(vpool.id(args))
        return var_cache[args]

    encodings_list: list[NNF] = []

    def append_formula(formula: NNF | list[NNF]) -> None:
        nonlocal encodings_list
        if isinstance(formula, list):
            encodings_list.extend(formula)
        else:
            encodings_list.append(formula)

    def encode_sum(literals: list[Var], bound: typing.Literal[0, 1]) -> NNF:
        if bound == 0:
            return nnf.And([~lit for lit in literals])
        else:
            return nnf.And(
                [
                    nnf.Or([~literals[i], ~literals[j]])
                    for i in range(len(literals))
                    for j in range(i + 1, len(literals))
                ]
                + [nnf.Or(literals)]
            )

    def gen_LR(i: int) -> collections.abc.Iterator[int]:
        for j in range(i + 1, min(2 * i, size - 1) + 1):
            if j % 2 == 0:
                yield j

    def gen_RR(i: int) -> collections.abc.Iterator[int]:
        for j in range(i + 2, min(2 * i + 1, size) + 1):
            if j % 2 == 1:
                yield j

    def gen_P(i: int) -> collections.abc.Iterator[int]:
        for j in range(i // 2, i):
            yield j

    # (1)
    append_formula(~var("v", 1))

    # (2)
    append_formula(
        [
            implies(var("v", i), ~(var("l", i, j)))
            for i in range(1, size + 1)
            for j in gen_LR(i)
        ]
    )

    # (3)
    append_formula(
        [
            iff(var("l", i, j), var("r", i, j + 1))
            for i in range(1, size + 1)
            for j in gen_LR(i)
        ]
    )

    # (4)
    append_formula(
        [
            implies(
                ~var("v", i),
                encode_sum([var("l", i, j) for j in gen_LR(i)], 1),
            )
            for i in range(1, size + 1)
        ]
    )

    # (5)
    append_formula(
        [
            iff(var("p", j, i), var("l", i, j))
            for i in range(1, size + 1)
            for j in gen_LR(i)
        ]
    )
    append_formula(
        [
            iff(var("p", j, i), var("r", i, j + 1))
            for i in range(1, size + 1)
            for j in gen_RR(i)
        ]
    )

    # (6)
    append_formula(
        [encode_sum([var("p", j, i) for i in gen_P(j)], 1) for j in range(2, size + 1)]
    )

    # (7)
    for q in range(1, n_features + 1):
        append_formula(~var("d", q, 1, 0))
        append_formula(
            [
                iff(
                    var("d", q, j, 0),
                    nnf.Or(
                        [
                            inner
                            for i in gen_P(j)
                            for inner in (
                                (var("p", j, i) & var("d", q, i, 0)),
                                (var("a", q, i) & var("r", i, j)),
                            )
                        ]
                    ),
                )
                for j in range(2, size + 1)
            ]
        )

    # (8)
    for q in range(1, n_features + 1):
        append_formula(~var("d", q, 1, 1))
        append_formula(
            [
                iff(
                    var("d", q, j, 1),
                    nnf.Or(
                        [
                            inner
                            for i in gen_P(j)
                            for inner in (
                                (var("p", j, i) & var("d", q, i, 1)),
                                (var("a", q, i) & var("l", i, j)),
                            )
                        ]
                    ),
                )
                for j in range(2, size + 1)
            ]
        )

    # (9)
    append_formula(
        [
            implies((var("u", r, i) & var("p", j, i)), ~var("a", r, j))
            for j in range(1, size + 1)
            for r in range(1, n_features + 1)
            for i in gen_P(j)
        ]
    )
    append_formula(
        [
            iff(
                var("u", r, j),
                nnf.Or(
                    [(var("u", r, i) & var("p", j, i)) for i in gen_P(j)]
                    + [var("a", r, j)]
                ),
            )
            for j in range(1, size + 1)
            for r in range(1, n_features + 1)
        ]
    )

    # (10)
    append_formula(
        [
            implies(
                ~var("v", j),
                encode_sum([var("a", r, j) for r in range(1, n_features + 1)], 1),
            )
            for j in range(1, size + 1)
        ]
    )

    # (11)
    append_formula(
        [
            implies(
                var("v", j),
                encode_sum([var("a", r, j) for r in range(1, n_features + 1)], 0),
            )
            for j in range(1, size + 1)
        ]
    )

    # (12) (13)
    for q in range(n_samples):
        if labels[q] == 1:
            append_formula(
                [
                    implies(
                        var("v", j) & ~var("c", j),
                        nnf.Or(
                            [
                                var("d", r, j, features[q][r - 1])
                                for r in range(1, n_features + 1)
                            ]
                        ),
                    )
                    for j in range(1, size + 1)
                ]
            )
        else:
            append_formula(
                [
                    implies(
                        var("v", j) & var("c", j),
                        nnf.Or(
                            [
                                ~var("d", r, j, features[q][r - 1])
                                for r in range(1, n_features + 1)
                            ]
                        ),
                    )
                    for j in range(1, size + 1)
                ]
            )

    final_encoding = to_CNF(nnf.And(encodings_list), simplify=True)

    var_labels = {}
    for outer in final_encoding.children:
        for inner in outer.children:
            if type(inner.name) is nnf.Aux:
                name = inner.name
                var_labels[name] = vpool.id(name)
            else:
                var_labels[inner.name] = inner.name

    dimacs_str = nnf.dimacs.dumps(final_encoding, var_labels=var_labels, mode="cnf")
    encodings = CNF(from_string=dimacs_str)

    with Solver(name="glucose3", bootstrap_with=encodings) as solver:
        if not solver.solve():
            return None

        model = solver.get_model()

        left = numpy.array([0] * (size + 1), dtype=numpy.int32)
        right = numpy.array([0] * (size + 1), dtype=numpy.int32)
        node_feature = numpy.array([0] * (size + 1), dtype=numpy.int32)
        node_label = numpy.array(
            [NODE_LABEL_IRRELEVANT] * (size + 1), dtype=numpy.int32
        )
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
                q = int(var_args[1])
                j = int(var_args[2])
                node_feature[j] = q + 1
            elif var_args[0] == "c" and assignment:
                j = int(var_args[1])
                node_label[j] = NODE_LABEL_POSITIVE
            else:
                pass

        for i in range(1, size + 1):
            if node_feature[i] == 0 and node_label[i] == NODE_LABEL_IRRELEVANT:
                node_label[i] = NOTE_LABEL_NEGATIVE

        return DecisionTree(
            left=left, right=right, features=node_feature, labels=node_label
        )


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

        tree = _build_dt_from_fixed_size(features, labels, size, n_features)
        if tree is None:
            break

        last_tree = tree

    if last_tree is None and trusted_bound:
        assert False, f"Failed to build tree with auto-computed max_size={max_size}"

    return last_tree
