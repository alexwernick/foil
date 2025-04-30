import copy
import itertools
import math
import time
from functools import lru_cache
from typing import Any, NamedTuple

from foil.first_order_logic import (
    ArgType,
    Clause,
    Literal,
    Predicate,
    RuleBasedPredicate,
    Variable,
)


class FOIL:
    _DEFAULT_TYPE_EXTENSION_LIMIT = 8

    target_literal: Literal
    predicates: list[Predicate]
    background_knowledge: dict[str, set[tuple]]
    background_knowledge_indices: dict
    beam_width: int
    _max_clause_length: int
    _timeout_seconds: int
    rules: list[Clause]
    _type_extension_limit: dict[ArgType, int]

    class BeamItem(NamedTuple):
        clause: Clause
        old_positive_examples: list[dict[str, Any]]
        old_negative_examples: list[dict[str, Any]]
        non_extended_pos_covered: list[dict[str, Any]]

    class BestLiteral(NamedTuple):
        gain: float
        literal: Literal
        pos_covered: list[dict[str, Any]]
        neg_covered: list[dict[str, Any]]
        non_extended_pos_covered: list[dict[str, Any]]

    def __init__(
        self,
        target_literal: Literal,
        predicates: list[Predicate],
        background_knowledge: dict[str, set[tuple]],
        beam_width: int = 1,
        max_clause_length: int = 7,
        timeout_seconds: int = 60,
        type_extension_limit={},
    ):
        """
        Initialize the FOIL algorithm.
        :param target_literal: The target literal to learn rules for.
        :param predicates: A list of Predicate objects.
        :param background_knowledge: A dict mapping predicate names to sets of facts.
        """
        self.target_literal = target_literal
        self.predicates = predicates
        self.background_knowledge = background_knowledge
        self.background_knowledge_indices = self._build_background_knowledge_indices()
        self.beam_width = beam_width
        self._max_clause_length = max_clause_length
        self._timeout_seconds = timeout_seconds
        self._type_extension_limit = type_extension_limit
        self.rules = []

    def fit(self, examples: list[tuple[bool, dict[str, Any]]]):
        """
        Learn rules to cover positive examples.
        :param examples: A list of tuples (is_positive, example)
        """
        start_time = time.time()
        # Separate positive and negative examples
        pos_examples: list[dict[str, Any]] = [ex for is_pos, ex in examples if is_pos]
        neg_examples: list[dict[str, Any]] = [
            ex for is_pos, ex in examples if not is_pos
        ]

        uncovered_pos = pos_examples.copy()

        while uncovered_pos:
            # Learn a new clause to cover positive examples
            clause, best_pos_covered = self._new_clause(
                uncovered_pos, neg_examples, start_time
            )
            # Remove positive examples covered by this clause
            uncovered_pos = [d for d in uncovered_pos if d not in best_pos_covered]
            self.rules.append(clause)

    def predict(self, example: dict[str, Any]) -> bool:
        """
        Predict if the example is positive based on learned rules.
        """
        for rule in self.rules:
            if rule.covers([example], self.background_knowledge):
                return True
        return False

    def _new_clause(
        self,
        uncovered_pos_examples: list[dict[str, Any]],
        neg_examples: list[dict[str, Any]],
        start_time: float,
    ) -> tuple[Clause, list[dict[str, Any]]]:
        """
        Learn a new clause to cover positive examples.
        :param uncovered_pos_examples: A list of positive examples not yet covered.
        :param neg_examples: A list of negative examples.
        :return: A new Clause object.
        """
        old_positive_examples = copy.deepcopy(uncovered_pos_examples)
        old_negative_examples = copy.deepcopy(neg_examples)

        # Start with a beam with a clause with only the head (the target literal)
        beam: list[FOIL.BeamItem] = [
            FOIL.BeamItem(
                clause=Clause(self.target_literal),
                old_positive_examples=old_positive_examples,
                old_negative_examples=old_negative_examples,
                non_extended_pos_covered=old_positive_examples,
            )
        ]

        while not self._beam_contains_item_with_no_negatives(
            beam
        ):  # while negative examples are covered
            new_beam: list[FOIL.BeamItem] = []

            # We check clause length. Fine to just check first
            # item as they all have the same length
            if len(beam[0].clause.body) >= self._max_clause_length:
                raise Exception(
                    f"Could not find a valid clause: max_clause_length {self._max_clause_length} reached"  # noqa: E501
                )

            for beam_item in beam:
                best_literals = self._find_next_best_literals(
                    beam_item.clause,
                    uncovered_pos_examples,
                    beam_item.old_positive_examples,
                    beam_item.old_negative_examples,
                    start_time,
                )

                for best_literal in best_literals:
                    clause = beam_item.clause.copy()
                    clause.add_literal(best_literal.literal)
                    new_beam.append(
                        FOIL.BeamItem(
                            clause=clause,
                            old_positive_examples=best_literal.pos_covered,
                            old_negative_examples=best_literal.neg_covered,
                            non_extended_pos_covered=best_literal.non_extended_pos_covered,  # noqa: E501
                        )
                    )

            if len(new_beam) == 0:
                raise Exception("Could not find a valid clause: no new literals found")

            # sort literals info gain (descending)
            new_beam = sorted(
                new_beam,
                key=lambda beam_item: self._information_gain(
                    len(beam_item.non_extended_pos_covered),
                    len(beam_item.old_positive_examples),
                    len(beam_item.old_negative_examples),
                    len(uncovered_pos_examples),
                    len(neg_examples),
                ),
                reverse=True,
            )

            # take top beam_width items
            beam = new_beam[: self.beam_width]

        beam = [
            beam_item for beam_item in beam if len(beam_item.old_negative_examples) == 0
        ]
        # We may have more than one beam with no negative examples
        # so we take the one with the highest information gain (sorted above)
        return (beam[0].clause, beam[0].non_extended_pos_covered)

    def _find_next_best_literals(
        self,
        clause: Clause,
        uncovered_pos_examples: list[dict[str, Any]],
        old_positive_examples: list[dict[str, Any]],
        old_negative_examples: list[dict[str, Any]],
        start_time: float,
    ) -> list[BestLiteral]:
        candidate_literals: list[Literal] = self._new_literals(clause)
        best_gain: float = 0.0
        best_literals: list[FOIL.BestLiteral] = []

        # Evaluate each candidate literal
        while candidate_literals:
            if time.time() - start_time > self._timeout_seconds:
                raise Exception(f"Timeout of {self._timeout_seconds} hit")

            literal = candidate_literals.pop(0)
            if literal in clause.body:
                continue  # Avoid cycles

            if literal == clause.head:
                continue  # Avoid recursion

            new_positive_examples: list[dict[str, Any]] = []
            old_positive_examples_copy = copy.deepcopy(old_positive_examples)

            for example in old_positive_examples_copy:
                new_positive_examples.extend(self._extend_example(example, literal))

            if literal.negated:
                new_positive_examples = self._trim_examples_with_duplicate_literals(
                    clause, new_positive_examples, literal
                )
                new_positive_examples_unextend = self._un_extend_examples(
                    old_positive_examples, new_positive_examples, literal.negated
                )
                new_positive_examples = []
                for example in copy.deepcopy(new_positive_examples_unextend):
                    new_positive_examples.extend(
                        self._extend_example(example, literal, evaluate_literal=False)
                    )

            new_positive_examples = self._trim_examples_with_duplicate_literals(
                clause, new_positive_examples, literal
            )

            new_non_extended_positive_examples = self._un_extend_examples(
                uncovered_pos_examples, new_positive_examples
            )

            new_non_extended_positive_examples_count = len(
                new_non_extended_positive_examples
            )

            new_positive_count = len(new_positive_examples)
            old_positive_count = len(old_positive_examples)
            old_negative_count = len(old_negative_examples)

            if not new_positive_count:
                more_specilaised_literals = {
                    Literal(pre, literal.args)
                    for pre in literal.predicate.more_specialised_predicates
                }
                candidate_literals = [
                    lit
                    for lit in candidate_literals
                    if lit not in more_specilaised_literals
                ]
                continue  # Must cover some positive examples

            # Now compute the maximum possible gain from substitutions
            # so we can prune. Max gain is assuming that we will have no
            # negative examples covered by the new literal
            max_gain = self._information_gain(
                new_non_extended_positive_examples_count,
                new_positive_count,
                0,
                old_positive_count,
                old_negative_count,
            )

            # If max_gain is less than best_gain, prune and
            # avoid extending negative examples
            if max_gain < best_gain:
                continue

            new_negative_examples: list[dict[str, Any]] = []
            old_negative_examples_copy = copy.deepcopy(old_negative_examples)
            for example in old_negative_examples_copy:
                new_negative_examples.extend(self._extend_example(example, literal))

            if literal.negated:
                new_negative_examples = self._trim_examples_with_duplicate_literals(
                    clause, new_negative_examples, literal
                )
                new_negative_examples_unextend = self._un_extend_examples(
                    old_negative_examples, new_negative_examples, literal.negated
                )
                new_negative_examples = []
                for example in copy.deepcopy(new_negative_examples_unextend):
                    new_negative_examples.extend(
                        self._extend_example(example, literal, evaluate_literal=False)
                    )

            new_negative_examples = self._trim_examples_with_duplicate_literals(
                clause, new_negative_examples, literal
            )

            new_negative_count = len(new_negative_examples)

            gain = self._information_gain(
                new_non_extended_positive_examples_count,
                new_positive_count,
                new_negative_count,
                old_positive_count,
                old_negative_count,
            )

            if gain > best_gain:
                # We append to our list of best literals
                best_literals.append(
                    FOIL.BestLiteral(
                        gain=gain,
                        literal=literal,
                        pos_covered=new_positive_examples,
                        neg_covered=new_negative_examples,
                        non_extended_pos_covered=new_non_extended_positive_examples,
                    )
                )

                # no point having more best literals than beam width
                if len(best_literals) > self.beam_width:
                    # Sort the list (ascending order)
                    best_literals = sorted(best_literals, key=lambda lit: lit.gain)
                    # Remove the worst
                    best_literals.pop(0)
                    # we then set gain below to be the
                    # worst value of the best literals (first item in list)
                    best_gain = best_literals[0].gain

        return best_literals

    def _extend_example(
        self,
        example: dict[str, Any],
        literal_to_add: Literal,
        evaluate_literal: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Extend the example with the target predicate.
        """
        # example will be like (true, {V1: 1, V2: 2})
        # literal will be like Q(V1, V3)
        # need to loop over possible values
        # for V3 and add to examples if it satisfies the literal
        extended_examples: list[dict[str, Any]] = [example]

        new_vars: list[Variable] = [
            arg for arg in literal_to_add.args if arg.name not in example
        ]

        return self._extend_example_with_new_vars(
            new_vars, extended_examples, literal_to_add, evaluate_literal
        )

    def _extend_example_with_new_vars(
        self,
        new_vars: list[Variable],
        extended_examples: list[dict[str, Any]],
        literal_to_add: Literal,
        evaluate_literal: bool,
    ) -> list[dict[str, Any]]:
        valid_extended_examples: list[dict[str, Any]] = []

        if len(new_vars) > 1:
            new_extended_examples: list[dict[str, Any]] = []
            new_var = new_vars.pop()
            for example in extended_examples:
                for value in new_var.arg_type.possible_values(example):
                    example[new_var.name] = value
                    if not evaluate_literal or self._partial_evaluate_literal(
                        literal_to_add, example
                    ):
                        new_extended_examples.append(copy.copy(example))

            return self._extend_example_with_new_vars(
                new_vars, new_extended_examples, literal_to_add, evaluate_literal
            )

        if len(new_vars) == 1:
            new_var = new_vars.pop()
            for example in extended_examples:
                for value in new_var.arg_type.possible_values(example):
                    example[new_var.name] = value
                    if not evaluate_literal or self._evaluate_literal(
                        literal_to_add, example
                    ):
                        valid_extended_examples.append(copy.copy(example))
            return valid_extended_examples

        for example in extended_examples:
            if not evaluate_literal or self._evaluate_literal(literal_to_add, example):
                valid_extended_examples.append(example)  # no new variables to add
        return valid_extended_examples

    def _new_literals_for_predicate(
        self,
        predicate: Predicate,
        clause: Clause,
        allow_variable_extension: bool = True,
    ) -> list[Literal]:
        """
        Generate possible literals to add to a clause's body.
        :param rule: The current rule being specialized.
        :return: A list of Literal objects.
        """
        possible_literals: list[Literal] = []
        # get the current variables in the clause
        current_vars = list(clause.variables)

        valid_vars_per_arg: list[list[Variable]] = [[] for _ in range(predicate.arity)]

        for var in current_vars:
            for i, arg_type in enumerate(predicate.arg_types):
                if arg_type == var.arg_type:
                    valid_vars_per_arg[i].append(var)

        var_numbers: list[int] = []
        for var in current_vars:
            var_numbers.append(int(var.name[1:]))

        current_max_var_num = max(var_numbers)

        if allow_variable_extension:
            # Introduce new variables (e.g., V1, V2)
            for arg_type, valid_vars in zip(predicate.arg_types, valid_vars_per_arg):
                extension_limit = self._type_extension_limit.get(
                    arg_type, self._DEFAULT_TYPE_EXTENSION_LIMIT
                )
                current_count = len([v for v in current_vars if v.arg_type == arg_type])

                if current_count >= extension_limit:
                    continue
                current_max_var_num = current_max_var_num + 1
                valid_vars.append(Variable(f"V{current_max_var_num}", arg_type))

        # Generate all combinations of existing variables
        var_combinations = [
            combo
            for combo in itertools.product(*valid_vars_per_arg)
            if len(set(combo)) == len(combo)  # avoid duplicates
        ]

        var_combinations_linked_to_clause = []
        for var_combo in var_combinations:
            if len(set(var_combo).intersection(clause.variables)) > 0:
                var_combinations_linked_to_clause.append(var_combo)

        # Create literals for each variable combination
        for vars in var_combinations_linked_to_clause:
            variables = list(vars)
            literal = Literal(predicate, variables)
            if literal in clause.incompatible_literals:
                continue

            negated_literal = Literal(predicate, variables, negated=True)

            # Avoid adding duplicate literals
            # We compare this way and purposefully don't check negation
            # i.e. if we have Q(V1, V2) we don't want to add not Q(V1, V2)
            lit_in_clause = False
            for lit in clause.body:
                if lit.predicate == predicate and literal.args == lit.args:
                    lit_in_clause = True
                    break

            if lit_in_clause:
                continue

            possible_literals.append(literal)
            if predicate.allow_negation:
                possible_literals.append(negated_literal)

        return possible_literals

    def _new_literals(
        self, clause: Clause, allow_variable_extension: bool = True
    ) -> list[Literal]:
        """
        Generate new literals to add to the clause.
        """
        literals: list[Literal] = []
        for predicate in self.predicates:
            # Generate possible literals
            literals.extend(
                self._new_literals_for_predicate(
                    predicate, clause, allow_variable_extension
                )
            )

        return literals

    @staticmethod
    def _information_gain(
        new_non_extended_positive_examples_count: int,
        new_positive_count: int,
        new_negative_count: int,
        old_positive_count: int,
        old_negative_count: int,
    ) -> float:
        """
        Calculate the adjusted FOIL information gain when introducing new literals.
        """
        # Avoid division by zero and log of zero
        if (
            new_non_extended_positive_examples_count == 0
            or new_positive_count == 0
            or old_positive_count == 0
        ):
            return 0

        # Compute probabilities
        prob_old = old_positive_count / (old_positive_count + old_negative_count)
        prob_new = new_positive_count / (new_positive_count + new_negative_count)

        # Calculate information
        information_old = -math.log2(prob_old)
        information_new = -math.log2(prob_new)

        # Compute information gain
        gain = new_non_extended_positive_examples_count * (
            information_old - information_new
        )
        return gain

    @staticmethod
    def _un_extend_examples(
        old_examples: list[dict[str, Any]],
        new_examples: list[dict[str, Any]],
        negated_literal: bool = False,
    ) -> list[dict[str, Any]]:
        is_example_covered = True
        non_extended_examples: list[dict[str, Any]] = []
        for old_example in old_examples:
            for new_posivive in new_examples:
                is_example_covered = True
                for key in old_example.keys():
                    if old_example[key] != new_posivive[key]:
                        is_example_covered = False
                        break
                if is_example_covered:
                    break

            if (is_example_covered and not negated_literal) or (
                not is_example_covered and negated_literal
            ):
                non_extended_examples.append(old_example)

        return non_extended_examples

    def _beam_contains_item_with_no_negatives(
        self, beam: list["FOIL.BeamItem"]
    ) -> bool:
        for item in beam:
            if not item.old_negative_examples:
                return True
        return False

    def _evaluate_literal(
        self,
        literal: Literal,
        example: dict[str, Any],
    ) -> bool:
        """
        Evaluate if a literal is satisfied by an example using background facts.
        :param literal: The Literal object to evaluate.
        :param example: A dict mapping variable names to values.
        :param background_knowledge: A dict mapping predicate names to sets of facts.
        :return: True if the literal is satisfied, False otherwise.
        """
        # Bind the arguments using the example's variable assignments
        bound_args = []
        for arg in literal.args:
            if arg.name in example:
                bound_args.append(example[arg.name])
                continue

            raise ValueError(f"Unbound variable '{arg}' in literal '{literal}'")

        # If it's rule based we use rule and don't evaluate background knowledge
        if isinstance(literal.predicate, RuleBasedPredicate):
            return literal.predicate.evaluate(*bound_args)

        fact: bool = self._is_a_fact(literal.predicate.name, tuple(bound_args))

        return fact

    @lru_cache(maxsize=5000000)
    def _is_a_fact(self, predicate_name: str, bound_args: tuple) -> bool:
        # Retrieve the predicate's facts
        predicate_facts = self.background_knowledge.get(predicate_name, set())
        return bound_args in predicate_facts

    def _partial_evaluate_literal(
        self,
        literal: Literal,
        example: dict[str, Any],
    ) -> bool:
        """
        Evaluate if literal can ever be satisfied by an example
        :param literal: The Literal object to evaluate.
        :param example: A dict mapping variable names to values.
        :param background_knowledge: A dict mapping predicate names to sets of facts.
        :return: True if the literal is satisfied, False otherwise.
        """
        bound_args = [
            example[arg.name] if arg.name in example else None for arg in literal.args
        ]

        all_vars_bound = all(arg is not None for arg in bound_args)

        # If it's rule based we have to assume all good
        if isinstance(literal.predicate, RuleBasedPredicate):
            if not all_vars_bound:
                return True

            # should never reach here in reality
            return literal.predicate.evaluate(*bound_args)

        # Check if the bound arguments are in the predicate's facts
        possible_fact = self._is_a_possible_fact(
            literal.predicate.name, tuple(bound_args)
        )

        return possible_fact

    def _is_a_possible_fact(self, predicate_name: str, bound_args: tuple) -> bool:
        index = self.background_knowledge_indices.get(predicate_name)
        if not index:
            return False
        return self._search_index(index, bound_args, 0)

    def _search_index(self, index, bound_args, position, allow_none=True):
        if "__fact__" in index:
            return True
        if position >= len(bound_args):
            return False
        arg_value = bound_args[position]
        next_indices = []
        if arg_value is None:
            if not allow_none:
                raise ValueError("Unexpected None value in bound arguments")
            # Wildcard, explore all possible values at this position
            next_indices = index.values()
        else:
            if arg_value in index:
                next_indices = [index[arg_value]]
            else:
                return False  # No match possible
        for next_index in next_indices:
            if self._search_index(next_index, bound_args, position + 1):
                return True
        return False

    def _build_background_knowledge_indices(self) -> dict:
        bk_indices = {}
        for predicate_name, facts in self.background_knowledge.items():
            index: dict = {}
            for fact in facts:
                current_level = index
                for i, value in enumerate(fact):
                    if value not in current_level:
                        current_level[value] = {}
                    current_level = current_level[value]
                # Store the fact at the deepest level
                current_level["__fact__"] = fact
            bk_indices[predicate_name] = index
        return bk_indices

    @staticmethod
    def _trim_examples_with_duplicate_literals(
        clause: Clause, examples: list[dict[str, Any]], literal: Literal
    ) -> list[dict[str, Any]]:
        pairs: list[tuple[str, str]] = []
        for arg in literal.args:
            for var in clause.variables:
                if arg.arg_type == var.arg_type and arg.name != var.name:
                    pairs.append((arg.name, var.name))

        if not pairs:
            return examples

        examples_trimmed = []
        for example in examples:
            args_match = False
            for arg_name_1, arg_name_2 in pairs:
                if example[arg_name_1] == example[arg_name_2]:
                    args_match = True
                    break
            if not args_match:
                examples_trimmed.append(example)

        return examples_trimmed
