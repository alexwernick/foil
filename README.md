# FOIL: First-Order Inductive Learner

## Overview

This project implements the FOIL (First-Order Inductive Learner) algorithm, a classic method for learning logical rules from examples. FOIL is designed to learn Horn clauses that can be used to predict whether a given example is positive or negative. The implementation is modular and allows for customization of predicates, background knowledge, and search parameters. This version incorporates beam search to efficiently explore the space of possible clauses.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [File Descriptions](#file-descriptions)
- [Class Details](#class-details)
  - [FOIL.py](#foilpy)
  - [first_order_logic.py](#first_order_logicpy)
- [Beam Search Implementation](#beam-search-implementation)
- [Information Gain Calculation](#information-gain-calculation)
- [Extending Examples](#extending-examples)
- [Suggested Usage](#suggested-usage)
- [Example](#example)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Installation

To install the project, you need to have Python 3.12+ and Poetry installed. Follow these steps:

1.  Clone the repository:

    ```bash
    git clone https://github.com/alexwernick/foil.git
    cd foil
    ```

2.  Install dependencies using Poetry:

    ```bash
    poetry install
    ```

## File Descriptions

*   `FOIL.py`: Contains the implementation of the FOIL algorithm, including the beam search.
*   `first_order_logic.py`: Defines the classes for first-order logic concepts such as predicates, literals, clauses, and argument types.
*   `test_FOIL.py`: Includes unit tests for the FOIL algorithm and related classes.

## Class Details

### `FOIL.py`

This file contains the `FOIL` class, which is the core of the FOIL algorithm implementation.

*   **`FOIL` Class**:

    *   **`__init__(self, target_literal, predicates, background_knowledge, beam_width=1, max_clause_length=7, timeout_seconds=60, type_extension_limit={})`**:
        *   Initializes the FOIL learner.
        *   `target_literal` (Literal): The target literal to learn rules for. This is the predicate you are trying to define.
        *   `predicates` (list[Predicate]): A list of Predicate objects that can be used in the body of the learned clauses. These are your building blocks.
        *   `background_knowledge` (dict[str, set[tuple]]): A dictionary mapping predicate names to sets of facts. This is the data FOIL uses to evaluate literals.  Each key is a predicate name (string), and the value is a set of tuples representing the known true instances of that predicate.
        *   `beam_width` (int): The beam width for beam search (default: 1).  This controls how many candidate clauses are kept at each step of the search.  A higher beam width explores more possibilities but requires more memory and computation.
        *   `max_clause_length` (int): The maximum length of a clause (default: 7).  This limits the complexity of the learned rules.
        *   `timeout_seconds` (int): The maximum time in seconds to run the algorithm (default: 60).  This prevents the algorithm from running indefinitely.
        *   `type_extension_limit` (dict[ArgType, int]): Limits the number of variables of each type (default: {}). This prevents the algorithm from creating too many variables of a given type, which can lead to combinatorial explosion.
    *   **`fit(self, examples)`**:
        *   Learns rules from the given examples.  This is the main entry point for training the FOIL model.
        *   `examples` (list[tuple[bool, dict[str, Any]]]): A list of tuples where the first element is a boolean indicating whether the example is positive, and the second element is a dictionary representing the example.  The dictionary maps variable names to their values in the example.
    *   **`predict(self, example)`**:
        *   Predicts whether an example is positive based on the learned rules.
        *   `example` (dict[str, Any]): A dictionary representing the example, mapping variable names to values.
        *   Returns `True` if the example is predicted as positive, `False` otherwise.
    *   **Internal Methods**:
        *   `_new_clause(self, uncovered_pos_examples, neg_examples, start_time)`: Learns a new clause to cover positive examples.  This is the core learning loop, iteratively refining a clause until it covers a sufficient number of positive examples and few negative examples.
        *   `_find_next_best_literals(self, clause, uncovered_pos_examples, old_positive_examples, old_negative_examples, start_time)`: Finds the best literals to add to the current clause. This method generates candidate literals and evaluates them based on information gain.
        *   `_extend_example(self, example, literal_to_add, evaluate_literal=True)`: Extends an example with the target predicate. This method finds all possible values for the unbound variables in the literal, creating new examples for each possible binding.
        *   `_new_literals_for_predicate(self, predicate, clause, allow_variable_extension=True)`: Generates possible literals to add to a clause's body, considering existing variables and potentially introducing new ones.
        *   `_new_literals(self, clause, allow_variable_extension=True)`: Generates new literals to add to the clause by iterating through available predicates.
        *   `_information_gain(new_non_extended_positive_examples_count, new_positive_count, new_negative_count, old_positive_count, old_negative_count)`: Calculates the adjusted FOIL information gain. This is the heuristic used to evaluate the "goodness" of a literal.
        *   `_evaluate_literal(self, literal, example)`: Evaluates if a literal is satisfied by an example using background facts.
        *   `_partial_evaluate_literal(self, literal, example)`: Evaluates if a literal can ever be satisfied by an example. This is an optimization to avoid unnecessary computations.
        *   `_build_background_knowledge_indices(self)`: Builds indices for background knowledge to speed up evaluation. This creates a nested dictionary structure to quickly check if a fact is known.
        *   `_trim_examples_with_duplicate_literals(clause, examples, literal)`: Trims examples with duplicate literals. This is to avoid redundant or inconsistent information.

### `first_order_logic.py`

This file defines the classes for representing first-order logic concepts.

*   **`ArgType` Class**:
    *   Represents the type of an argument (e.g., "person", "node").
    *   `__init__(self, name: str, possible_values: Optional[list] = None, possible_values_fn: Optional[Callable[[dict[str, Any]], list]] = None)`: Initializes an ArgType.
        *   `name` (str): The name of the argument type.
        *   `possible_values` (Optional[list]): A list of possible values for the argument type.
        *   `possible_values_fn` (Optional[Callable[[dict[str, Any]], list]]): A function that returns possible values based on an example.
    *   `possible_values(self, example: dict[str, Any]) -> list`: Returns the possible values for the argument type based on the example.
*   **`Variable` Class**:
    *   Represents a variable in a logical expression (e.g., "X", "Y").
    *   `__init__(self, name: str, arg_type: ArgType)`: Initializes a Variable.
        *   `name` (str): The name of the variable.
        *   `arg_type` (ArgType): The argument type of the variable.
*   **`Predicate` Class**:
    *   Represents a predicate in first-order logic (e.g., "parent", "linked-to").
    *   `__init__(self, name: str, arity: int, arg_types: list[ArgType], incompatible_predicates: Optional[set["Predicate"]] = None, more_specialised_predicates: Optional[set["Predicate"]] = None, allow_negation: bool = False)`: Initializes a Predicate.
        *   `name` (str): The name of the predicate.
        *   `arity` (int): The number of arguments the predicate takes.
        *   `arg_types` (list[ArgType]): A list of argument types for the predicate.
        *   `incompatible_predicates` (Optional[set["Predicate"]]): A set of predicates that are incompatible with this predicate.
        *   `more_specialised_predicates` (Optional[set["Predicate"]]): A set of predicates that are more specialised than this predicate.
        *   `allow_negation` (bool): Boolean flag indicating if the predicate can be negated.
*   **`RuleBasedPredicate` Class**:
    *   Represents a predicate that is evaluated using a Python function.
    *   `__init__(self, name: str, arity: int, arg_types: list[ArgType], eval_fn: Callable[..., bool])`: Initializes a RuleBasedPredicate.
        *   `name` (str): The name of the predicate.
        *   `arity` (int): The number of arguments the predicate takes.
        *   `arg_types` (list[ArgType]): A list of argument types for the predicate.
        *   `eval_fn` (Callable[..., bool]): A function that evaluates the predicate.
    *   `evaluate(self, *args: Any) -> bool`: Evaluates the predicate with the given arguments.
*   **`Literal` Class**:
    *   Represents a literal in first-order logic, consisting of a predicate and its arguments.
    *   `__init__(self, predicate: Predicate, args: List[Variable], negated: bool = False)`: Initializes a Literal.
        *   `predicate` (Predicate): The Predicate object associated with this literal.
        *   `args` (List[Variable]): A list of arguments (variables or constants).
        *   `negated` (bool): Boolean flag indicating if the literal is negated.
*   **`Clause` Class**:
    *   Represents a clause in first-order logic, consisting of a head literal and a body of literals.
    *   `__init__(self, head: Literal)`: Initializes a Clause.
        *   `head` (Literal): The head of the clause (target literal).
    *   `add_literal(self, literal: Literal)`: Adds a literal to the clause's body.
    *   `covers(self, examples: list[dict[str, Any]], background_knowledge: dict[str, set[tuple]])`: Checks which examples are covered by this rule.

## Beam Search Implementation

The `FOIL` class uses beam search to efficiently explore the space of possible clauses. Here's how it works:

1.  **Initialization:** The search starts with a beam containing a single `BeamItem`. This `BeamItem` represents a clause with only the head literal (the target predicate). It also stores the positive and negative examples that are covered by this initial clause.

2.  **Iteration:** The algorithm iterates as long as the beam does not contain an item (clause) that covers no negative examples.

3.  **Literal Generation:** For each `BeamItem` in the current beam, the `_find_next_best_literals` method generates candidate literals to add to the clause. This involves considering all possible predicates and variable combinations.

4.  **Literal Evaluation:** Each candidate literal is evaluated based on the adjusted FOIL information gain. This metric measures how much the literal improves the clause's ability to discriminate between positive and negative examples. The information gain calculation is explained in detail below.

5.  **Beam Update:** The algorithm sorts the new `BeamItem` objects (clauses with added literals) based on their information gain. It then selects the top `beam_width` items to form the new beam. This ensures that the most promising clauses are kept for further refinement.

6.  **Termination:** The search terminates when a `BeamItem` is found that covers no negative examples. This clause is considered a good rule for predicting the target predicate.

The `BeamItem` namedtuple stores the following information for each candidate clause:

*   `clause`: The Clause object representing the current state of the clause.
*   `old_positive_examples`: The positive examples covered by the clause.
*   `old_negative_examples`: The negative examples covered by the clause.
*   `non_extended_pos_covered`: Positive examples covered by the clause without variable extension.

## Information Gain Calculation

The `_information_gain` method calculates the adjusted FOIL information gain, which is used to evaluate the "goodness" of a literal. The formula is:

```
Gain = new_non_extended_positive_examples_count * (information_old - information_new)
```

where:

*   `new_non_extended_positive_examples_count`: The number of positive examples covered by the new clause (with the added literal) that were also covered by the original clause, without extending the examples. This is a measure of how much the literal improves the coverage of positive examples.
*   `information_old`: The information content of the original clause, calculated as `-log2(prob_old)`, where `prob_old` is the proportion of positive examples covered by the original clause.
*   `information_new`: The information content of the new clause, calculated as `-log2(prob_new)`, where `prob_new` is the proportion of positive examples covered by the new clause.

The information gain measures how much the literal reduces the uncertainty in predicting the target predicate. A higher information gain indicates that the literal is more informative and useful for discriminating between positive and negative examples.

## Extending Examples

The `_extend_example` method plays a crucial role in FOIL by finding all possible values for unbound variables in a literal. This process is essential for discovering relationships between variables and making the learned rules more general.

Given an example and a literal, the `_extend_example` method does the following:

1.  **Identify Unbound Variables:** It identifies the variables in the literal that are not already assigned values in the example.

2.  **Find Possible Values:** For each unbound variable, it retrieves the possible values from the `ArgType` associated with the variable.

3.  **Create New Examples:** It creates new examples for each possible binding of the unbound variables. Each new example is a copy of the original example, with the unbound variables assigned specific values.

4.  **Evaluate the Literal:** If `evaluate_literal` is True (the default), it evaluates the literal with the new variable assignments. Only the examples that satisfy the literal are kept.

This process can generate multiple extended examples from a single original example, allowing FOIL to explore a wider range of possible relationships between variables.

## Suggested Usage

1.  **Define Argument Types**:

    *   Create `ArgType` instances for each type of argument your predicates will use.  Specify possible values or a function to generate them.

    ```python
    from foil.first_order_logic import ArgType

    node_type = ArgType("node", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    ```

2.  **Define Predicates**:

    *   Create `Predicate` instances for each predicate in your domain.  Specify the arity and argument types.

    ```python
    from foil.first_order_logic import Predicate

    linked_to_pred = Predicate("linked-to", 2, [node_type, node_type])
    ```

3.  **Define the Target Literal**:

    *   Create a `Literal` instance representing the target predicate you want to learn.

    ```python
    from foil.first_order_logic import Literal, Variable

    V1 = Variable("V1", node_type)
    V2 = Variable("V2", node_type)
    target_predicate = Predicate("can-reach", 2, [node_type, node_type])
    target_literal = Literal(predicate=target_predicate, args=[V1, V2])
    ```

4.  **Provide Background Knowledge**:

    *   Create a dictionary mapping predicate names to sets of facts.

    ```python
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
    ```

5.  **Prepare Examples**:

    *   Create a list of tuples, where each tuple contains a boolean (indicating positive or negative) and a dictionary representing the example.

    ```python
    positive_examples = {(0, 1), (0, 2), (0, 3)}  # Example positive examples
    examples = [
        ((i, j) in positive_examples, {V1.name: i, V2.name: j})
        for i in range(5)
        for j in range(5)
        if i != j
    ]
    ```

6.  **Instantiate and Run FOIL**:

    *   Create a `FOIL` instance with the target literal, predicates, and background knowledge.  Then, call the `fit` method with your examples.

    ```python
    from foil.FOIL import FOIL

    foil = FOIL(target_literal, [linked_to_pred], background_knowledge)
    foil.fit(examples)
    ```

7.  **Make Predictions**:

    *   Use the `predict` method to predict whether new examples are positive or negative.

    ```python
    new_example = {V1.name: 1, V2.name: 3}
    prediction = foil.predict(new_example)
    print(f"Prediction for {new_example}: {prediction}")
    ```

## Example

```python
from foil.first_order_logic import ArgType, Predicate, Literal, Variable
from foil.FOIL import FOIL

# Define argument type
node_type = ArgType("node", [0, 1, 2, 3])

# Define predicates
linked_to_pred = Predicate("linked-to", 2, [node_type, node_type])

# Define target literal
V1 = Variable("V1", node_type)
V2 = Variable("V2", node_type)
target_predicate = Predicate("can-reach", 2, [node_type, node_type])
target_literal = Literal(predicate=target_predicate, args=[V1, V2])

# Background knowledge
background_knowledge = {
    "linked-to": {
        (0, 1),
        (1, 2),
        (2, 3),
    }
}

# Prepare examples
positive_examples = {(0, 1), (1, 2), (2, 3), (0,2), (1,3), (0,3)}
examples = [
    ((i, j) in positive_examples, {V1.name: i, V2.name: j})
    for i in range(4)
    for j in range(4)
    if i != j
]

# Instantiate and run FOIL
foil = FOIL(target_literal, [linked_to_pred], background_knowledge)
foil.fit(examples)

# Make predictions
new_example = {V1.name: 0, V2.name: 3}
prediction = foil.predict(new_example)
print(f"Prediction for {new_example}: {prediction}")  # Expected: True

new_example = {V1.name: 3, V2.name: 0}
prediction = foil.predict(new_example)
print(f"Prediction for {new_example}: {prediction}") # Expected: False
```

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes and add unit tests.
4.  Submit a pull request.

## Acknowledgments

*   This project is inspired by the original FOIL algorithm described in the paper "Learning Logical Definitions from Relations" by Quinlan et al.