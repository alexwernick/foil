from foil.first_order_logic import (
    ArgType,
    Clause,
    Literal,
    Predicate,
    Variable,
)
from foil.FOIL import FOIL


def test_FOIL_with_simple_example():
    """
    Test FOIL algorithm
    This example is taken from the paper
    'Learning Logical Definitions from Relations' by Quinlan et al.
    The example is about learning a 'can-reach' from 'linked-to' relations.
    """
    node_type = ArgType("node", [0, 1, 2, 3, 4, 5, 6, 7, 8])  # 9 nodes

    target_predicate = Predicate("can-reach", 2, [node_type, node_type])

    V1 = Variable("V1", node_type)
    V2 = Variable("V2", node_type)

    target_literal = Literal(predicate=target_predicate, args=[V1, V2])

    # Available predicates
    predicates = [Predicate("linked-to", 2, [node_type, node_type])]

    # Positive examples
    positive_examples = {
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 8),
        (1, 2),
        (3, 2),
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 8),
        (4, 5),
        (4, 6),
        (4, 8),
        (6, 8),
        (7, 6),
        (7, 8),
    }
    examples = [
        ((i, j) in positive_examples, {V1.name: i, V2.name: j})
        for i in range(10)
        for j in range(10)
        if i != j
    ]

    # Background facts for predicates
    background_knowledge = {
        "linked-to": {
            (0, 1),
            (0, 3),
            (1, 2),
            (3, 2),
            (3, 4),
            (4, 5),
            (4, 6),
            (6, 8),
            (7, 6),
            (7, 8),
        }
    }

    foil = FOIL(target_literal, predicates, background_knowledge)
    foil.fit(examples)

    for example in examples:
        assert foil.predict(example[1]) == example[0]


def test_new_literals_for_predicate():
    node_type = ArgType("node", [0, 1, 2, 3, 4, 5, 6, 7, 8])  # 9 nodes
    another_type = ArgType("another", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    and_another_type = ArgType("and_another", [0, 1, 2, 3, 4, 5, 6, 7, 8])

    target_predicate = Predicate("can-reach", 2, [node_type, node_type])

    V1 = Variable("V1", node_type)
    V2 = Variable("V2", node_type)

    target_literal = Literal(predicate=target_predicate, args=[V1, V2])
    linked_to_pred = Predicate("linked-to", 2, [node_type, node_type])
    predicates = [linked_to_pred]
    foil = FOIL(target_literal, predicates, {})

    pre_with_one_arg = Predicate("one-arg", 1, [node_type])
    per_with_two_args = Predicate("two-arg", 2, [node_type, another_type])
    pre_with_three_arg = Predicate(
        "three-arg", 3, [node_type, another_type, and_another_type]
    )

    clause = Clause(target_literal)
    literals = foil._new_literals_for_predicate(per_with_two_args, clause)

    assert len(literals) == 2

    literals = foil._new_literals_for_predicate(pre_with_one_arg, clause)

    assert len(literals) == 2

    literals = foil._new_literals_for_predicate(pre_with_three_arg, clause)

    assert len(literals) == 2
