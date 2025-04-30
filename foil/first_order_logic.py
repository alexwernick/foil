import inspect
from collections import Counter
from typing import Any, Callable, List, Optional


class ArgType:
    name: str
    _possible_values: Optional[list]
    _possible_values_fn: Optional[Callable[[dict[str, Any]], list]]

    def __init__(
        self,
        name: str,
        possible_values: Optional[list] = None,
        possible_values_fn: Optional[Callable[[dict[str, Any]], list]] = None,
    ):
        """
        Initialize an ArgType.
        :param name: Name of the argument type (e.g., 'person').
        :possible_values: List of possible values for the argument type.
        :possible_values_fn: Function that returns possible values
        for the argument type based on an example.
        """
        if possible_values is None and possible_values_fn is None:
            raise ValueError(
                "Either possible_values or possible_values_fn must be provided"
            )

        self.name = name
        self._possible_values = possible_values
        self._possible_values_fn = possible_values_fn

    def __repr__(self):
        """
        String representation of the argument type.
        """
        return self.name

    def __eq__(self, other):
        """
        Equality check for comparing argument types.
        """
        return self.name == other.name

    def __hash__(self):
        """
        Hash function to allow argument types to be used in sets and as dictionary keys.
        """
        return hash(self.name)

    def possible_values(self, example: dict[str, Any]) -> list:
        if self._possible_values_fn:
            return self._possible_values_fn(example)

        if self._possible_values:
            return self._possible_values

        # this should never happen due to constructor validation
        raise ValueError("No possible values defined for this argument type")


class Variable:
    name: str
    arg_type: ArgType

    def __init__(self, name: str, arg_type: ArgType):
        """
        Initialize a Variable.
        :param name: The name of the variable (e.g., 'X', 'Y').
        """
        self.name = name  # String representing the variable's name
        self.arg_type = arg_type  # Possible values for the variable

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name

    def __hash__(self):
        return hash(("Variable", self.name))


class Predicate:
    name: str
    arity: int
    arg_types: list[ArgType]
    incompatible_predicates: set["Predicate"]
    more_specialised_predicates: set["Predicate"]
    allow_negation: bool

    def __init__(
        self,
        name: str,
        arity: int,
        arg_types: list[ArgType],
        incompatible_predicates: Optional[set["Predicate"]] = None,
        more_specialised_predicates: Optional[set["Predicate"]] = None,
        allow_negation: bool = False,
    ):
        """
        Initialize a Predicate.
        :param name: Name of the predicate (e.g., 'parent').
        :param arity: Number of arguments the predicate takes.
        """
        if len(arg_types) != arity:
            raise ValueError(f"Predicate '{name}' requires {arity} argument types")

        self.name = name
        self.arity = arity
        self.arg_types = arg_types
        incompatible_predicates = incompatible_predicates or set()
        self._validate_predicates_match(incompatible_predicates)
        self.incompatible_predicates = incompatible_predicates
        more_specialised_predicates = more_specialised_predicates or set()
        self._validate_predicates_match(more_specialised_predicates)
        self.more_specialised_predicates = more_specialised_predicates
        self.allow_negation = allow_negation

    def __repr__(self):
        """
        String representation of the predicate.
        """
        return f"{self.name}/{self.arity}"

    def __eq__(self, other):
        """
        Equality check for comparing predicates.
        """
        return self.name == other.name and self.arity == other.arity

    def __hash__(self):
        """
        Hash function to allow predicates to be used in sets and as dictionary keys.
        """
        return hash((self.name, self.arity, tuple(self.arg_types)))

    def add_incompatible_predicate(self, inompatible_predicate: "Predicate"):
        self._validate_predicate_matches(inompatible_predicate)
        self.incompatible_predicates.add(inompatible_predicate)

    def add_incompatible_predicates(self, inompatible_predicates: set["Predicate"]):
        self._validate_predicates_match(inompatible_predicates)
        self.incompatible_predicates.update(inompatible_predicates)

    def add_more_specialised_predicate(self, more_specialised_predicate: "Predicate"):
        self._validate_predicate_matches(more_specialised_predicate)
        self.incompatible_predicates.add(more_specialised_predicate)

    def add_more_specialised_predicates(
        self, more_specialised_predicates: set["Predicate"]
    ):
        self._validate_predicates_match(more_specialised_predicates)
        self.incompatible_predicates.update(more_specialised_predicates)

    def _validate_predicates_match(self, predicates: set["Predicate"]):
        for p in predicates:
            self._validate_predicate_matches(p)

    def _validate_predicate_matches(self, predicate: "Predicate"):
        if predicate.arity != self.arity:
            raise ValueError(
                f"Predicate '{self.name}' incompatible with '{predicate.name}' as they have different arity"  # noqa: E501
            )
        if predicate.arg_types != self.arg_types:
            raise ValueError(
                f"Predicate '{self.name}' incompatible with '{predicate.name}' as they have different argument types"  # noqa: E501
            )


class RuleBasedPredicate(Predicate):
    eval_fn: Callable[..., bool]

    def __init__(
        self,
        name: str,
        arity: int,
        arg_types: list[ArgType],
        eval_fn: Callable[..., bool],
    ):
        super().__init__(name, arity, arg_types)

        signature = inspect.signature(eval_fn)
        parameters = signature.parameters
        if len(parameters) != arity:
            raise ValueError(
                f"Predicate '{name}' requires eval_fn to have arity: {arity} parameters"
            )

        self.eval_fn = eval_fn

    def evaluate(self, *args: Any) -> bool:
        return self.eval_fn(*args)


class Literal:
    predicate: Predicate
    args: List[Variable]
    negated: bool

    def __init__(
        self, predicate: Predicate, args: List[Variable], negated: bool = False
    ):
        """
        Initialize a Literal.
        :param predicate: The Predicate object associated with this literal.
        :param args: A list of arguments (variables or constants).
        :param negated: Boolean flag indicating if the literal is negated.
        """
        if len(args) != predicate.arity:
            raise ValueError(
                f"Predicate '{predicate.name}' requires {predicate.arity} arguments"
            )

        for arg, arg_type in zip(args, predicate.arg_types):
            if arg.arg_type != arg_type:
                raise ValueError(f"Variable '{arg.name}' must be of type '{arg_type}'")

        self.predicate = predicate
        self.args = args  # List of variables
        self.negated = negated

    def __repr__(self):
        """
        String representation of the literal.
        """
        args_str = ", ".join(arg.name for arg in self.args)
        negation = "not " if self.negated else ""
        return f"{negation}{self.predicate.name}({args_str})"

    def __eq__(self, other):
        """
        Equality check for comparing literals.
        """
        return (
            self.predicate == other.predicate
            and self.args == other.args
            and self.negated == other.negated
        )

    def __hash__(self):
        """
        Hash function to allow literals to be used in sets and as dictionary keys.
        """
        return hash((self.predicate, tuple(self.args), self.negated))


class Clause:
    head: Literal
    body: List[Literal]
    variables: set[Variable]
    incompatible_literals: set[Literal]

    def __init__(self, head: Literal):
        """
        Initialize a Rule.
        :param head_literal: The head of the rule (target literal).
        """
        self.head = head  # The target predicate as a Literal object
        self.body = []  # List of Literal objects in the rule's body
        self.variables = set()  # Variables used in the rule
        self.incompatible_literals = (
            set()
        )  # Literals that are incompatible with this clause

        for arg in head.args:
            # if isinstance(arg, Variable):
            self.variables.add(arg)  # Variables used in the rule

    def arg_type_var_names(self) -> dict[ArgType, list[str]]:
        return {
            arg_type: [var.name for var in self.variables if var.arg_type == arg_type]
            for arg_type in {var.arg_type for var in self.variables}
        }

    def add_literal(self, literal: Literal):
        """
        Add a literal to the rule's body.
        :param literal: The Literal object to add.
        """
        self.body.append(literal)
        # Update the list of variables with variables from the new literal
        self.variables.update(
            {arg for arg in literal.args}  # if isinstance(arg, Variable)
        )

        for incompatible_predicate in literal.predicate.incompatible_predicates:
            # We can do this as we have restricted the arguments to be the same
            # See the the Predicate class
            incompatible_literal = Literal(incompatible_predicate, literal.args)
            self.incompatible_literals.add(incompatible_literal)

    def copy(self) -> "Clause":
        """
        Create a copy of the clause.
        """
        new_clause = Clause(self.head)
        new_clause.body = self.body.copy()
        new_clause.variables = self.variables.copy()
        new_clause.incompatible_literals = self.incompatible_literals.copy()
        return new_clause

    def covers(
        self,
        examples: list[dict[str, Any]],
        background_knowledge: dict[str, set[tuple]],
    ):
        """
        Check which examples are covered by this rule.
        :param examples: A list of examples (dicts mapping variable names to values).
        :return: A list of examples covered by the rule.
        """
        covered = []
        for ex in examples:
            if self._is_satisfied(ex, background_knowledge):
                covered.append(ex)
        return covered

    def _is_satisfied(
        self, example: dict[str, Any], background_knowledge: dict[str, set[tuple]]
    ):
        """
        Check if the example satisfies the rule, possibly
        finding assignments for free variables.
        :param example: A dict mapping Variables to values (from the example).
        :return: True if the rule is satisfied for the given example, False otherwise.
        """
        # Evaluate each literal in the body with the given example
        variable_assignments = example.copy()

        # get the args that can't all be equal
        arg_type_var_names = self.arg_type_var_names()

        # Attempt to satisfy the body literals with possible bindings
        return self._satisfy_body(
            variable_assignments, background_knowledge, arg_type_var_names, 0
        )

    @staticmethod
    def _get_possible_bindings(
        literal: Literal,
        variable_assignments: dict[str, Any],
        background_knowledge: dict[str, set[tuple]],
        arg_type_var_names: dict[ArgType, list[str]],
    ) -> list[dict[str, Any]]:
        """
        Find possible variable bindings for a literal given current assignments.
        :param literal: The Literal object to satisfy.
        :param variable_assignments: Current assignments for variables (dict).
        :return: A list of possible new bindings (list of dicts).
        """
        # Prepare the arguments with current assignments, identify unbound variables
        args: list[Any] = []
        for arg in literal.args:
            if arg.name in variable_assignments:
                args.append(variable_assignments[arg.name])
            else:
                args.append(None)

        # If it's rule based we use rule and don't evaluate background knowledge
        if isinstance(literal.predicate, RuleBasedPredicate):
            return Clause._generate_possible_bindings_for_rule_based_predicate(
                args, literal, variable_assignments, arg_type_var_names
            )

        return Clause._generate_possible_bindings_for_predicate(
            args,
            literal,
            variable_assignments,
            arg_type_var_names,
            background_knowledge,
        )

    @staticmethod
    def _is_valid_value(
        value: Any,
        example: dict[str, Any],
        arg_type: ArgType,
        arg_type_var_names: dict[ArgType, list[str]],
    ) -> bool:
        """
        Check if the value is valid for the given argument type
        and the current assignments the mathing arg types
        """
        if arg_type not in arg_type_var_names:
            return True

        for var_name in arg_type_var_names[arg_type]:
            if var_name in example and example[var_name] == value:
                return False
        return True

    @staticmethod
    def _extend_args(
        args: list[Any],
        literal: Literal,
        variable_assignments: dict[str, Any],
        arg_type_var_names: dict[ArgType, list[str]],
    ) -> list[list[Any]]:
        extended_args = [args.copy()]
        for i, (arg_val, arg) in enumerate(zip(args, literal.args)):
            if arg_val is None:
                new_extended_args = []
                for ex_arg in extended_args:
                    for possible_val in arg.arg_type.possible_values(
                        variable_assignments
                    ):
                        if not Clause._is_valid_value(
                            possible_val,
                            variable_assignments,
                            arg.arg_type,
                            arg_type_var_names,
                        ):
                            continue

                        copy = ex_arg.copy()
                        copy[i] = possible_val
                        new_extended_args.append(copy)
                extended_args = new_extended_args
        return extended_args

    @staticmethod
    def _generate_possible_bindings_for_rule_based_predicate(
        args: list[Any],
        literal: Literal,
        variable_assignments: dict[str, Any],
        arg_type_var_names: dict[ArgType, list[str]],
    ) -> list[dict[str, Any]]:
        def generate_bindings(
            extended_args: list[list[Any]], evaluate_literal: bool
        ) -> list[dict[str, Any]]:
            if not isinstance(literal.predicate, RuleBasedPredicate):
                raise ValueError("Predicate must be a RuleBasedPredicate")
            possible_bindings: list[dict[str, Any]] = []
            for ex_args in extended_args:
                new_bindings = {}

                result: bool = True
                if evaluate_literal:
                    result = literal.predicate.evaluate(*ex_args)

                if result:
                    for arg_val, arg, ex_arg in zip(args, literal.args, ex_args):
                        if arg_val is None:
                            new_bindings[arg.name] = ex_arg
                    possible_bindings.append(new_bindings)
            return possible_bindings

        extended_args = Clause._extend_args(
            args, literal, variable_assignments, arg_type_var_names
        )

        if not isinstance(literal.predicate, RuleBasedPredicate):
            raise ValueError("Predicate must be a RuleBasedPredicate")

        possible_bindings = generate_bindings(extended_args, True)

        if literal.negated and len(possible_bindings) == 0:
            # This is the case where the negation is valid
            # hence we extend with all possible bindings
            possible_bindings = generate_bindings(extended_args, False)
            pass

        return possible_bindings

    @staticmethod
    def _generate_possible_bindings_for_predicate(
        args: list[Any],
        literal: Literal,
        variable_assignments: dict[str, Any],
        arg_type_var_names: dict[ArgType, list[str]],
        background_knowledge: dict[str, set[tuple]],
    ) -> list[dict[str, Any]]:
        def generate_bindings_for_negated(
            extended_args: list[list[Any]], pred_facts: set[tuple]
        ) -> list[dict[str, Any]]:
            possible_bindings: list[dict[str, Any]] = []
            none_true: bool = True
            for ex_args in extended_args:
                if tuple(ex_args) in pred_facts:
                    none_true = False
                    break

            if not none_true:
                return possible_bindings

            for ex_args in extended_args:
                new_bindings = {}
                for arg_val, arg, ex_arg in zip(args, literal.args, ex_args):
                    if arg_val is None:
                        new_bindings[arg.name] = ex_arg
                possible_bindings.append(new_bindings)
            return possible_bindings

        possible_bindings: list[dict[str, Any]] = []
        # Retrieve facts for the predicate
        predicate_facts = background_knowledge.get(literal.predicate.name, set())

        if not literal.negated:
            for fact in predicate_facts:
                # Check if the fact matches the known assignments
                match = True
                new_bindings = {}

                for arg_val, fact_val, arg in zip(args, fact, literal.args):
                    if arg_val is None:
                        # Unbound variable; propose a new binding
                        if not Clause._is_valid_value(
                            fact_val,
                            variable_assignments,
                            arg.arg_type,
                            arg_type_var_names,
                        ):
                            match = False
                            break  # No need to check further

                        new_bindings[arg.name] = fact_val

                    elif arg_val != fact_val:
                        # Known variable assignment does not match the fact's value
                        match = False
                        break  # No need to check further

                if match:
                    possible_bindings.append(new_bindings)
        else:
            extended_args = Clause._extend_args(
                args, literal, variable_assignments, arg_type_var_names
            )
            possible_bindings = generate_bindings_for_negated(
                extended_args, predicate_facts
            )

        return possible_bindings

    def _satisfy_body(
        self,
        variable_assignments: dict[str, Any],
        background_knowledge: dict[str, set[tuple]],
        arg_type_var_names: dict[ArgType, list[str]],
        literal_index: int,
    ):
        """
        Check if the example satisfies the rule,
        possibly finding assignments for free variables.
        :param example: A dict mapping Variables to values (from the example).
        :return: True if the rule is satisfied for the given example, False otherwise.
        """
        # Base case: All literals have been satisfied
        if literal_index >= len(self.body):
            return True

        literal = self.body[literal_index]

        # Get possible bindings for the current literal
        possible_bindings: list[dict[str, Any]] = self._get_possible_bindings(
            literal, variable_assignments, background_knowledge, arg_type_var_names
        )

        for binding in possible_bindings:
            # Update variable assignments with new bindings
            new_assignments = variable_assignments.copy()
            new_assignments.update(binding)

            # Recursive call to satisfy the next literal
            if self._satisfy_body(
                new_assignments,
                background_knowledge,
                arg_type_var_names,
                literal_index + 1,
            ):
                return True  # Found satisfying assignments for all literals

        return False  # No satisfying assignments found for this literal

    def _find_duplicate_predicates(self):
        preds = [lit.predicate for lit in self.body]
        counts = Counter(preds)
        return [item for item, count in counts.items() if count > 1]

    @staticmethod
    def _un_extend_examples(
        old_possible_bindings: list[dict[str, Any]],
        new_possible_bindings: list[dict[str, Any]],
        negated_literal: bool = False,
    ) -> list[dict[str, Any]]:
        is_example_covered = True
        non_extended_possible_bindings: list[dict[str, Any]] = []
        for old_example in old_possible_bindings:
            for new_posivive in new_possible_bindings:
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
                non_extended_possible_bindings.append(old_example)

        return non_extended_possible_bindings
