import os
import random
from qubots.base_problem import BaseProblem

# Define constants for objective order
COLOR_HIGH_LOW = 0
HIGH_LOW_COLOR = 1
HIGH_COLOR_LOW = 2
COLOR_HIGH = 3
HIGH_COLOR = 4

class CarSequencingColorProblem(BaseProblem):
    """
    Car Sequencing with Paint-Shop Batching Constraints

    In this problem, a sequence of cars (each belonging to a certain class) must be arranged.
    Each class is characterized by:
      - A color (an integer value).
      - A set of option requirements (a Boolean list for each option).

    The instance data includes:
      - nb_positions: number of cars (positions in the sequence)
      - nb_options: number of options
      - nb_classes: number of classes
      - paint_batch_limit: maximum number of consecutive cars without a paint change
      - objective_order: a code indicating the lexicographic order of objectives
      - start_position: positions before this value must remain fixed
      - For each option: max_cars_per_window (maximum allowed cars with that option in any window), window_size, and a flag indicating if the option is high priority.
      - For each class: color (an integer), number of cars in the class, and for each option a binary indicator (1 if the option is required, 0 otherwise).
      - initial_sequence: a list of length nb_positions giving the initial production plan (each entry is a class index).

    Candidate Solution:
      A permutation (list) of integers of length nb_positions that rearranges the initial production plan.
      Positions before start_position must remain fixed.
      
    Objective:
      Three objectives are computed:
        - objective_color: the total number of color changes between consecutive cars (from start_position onward).
        - objective_high_priority: total violations over all high-priority options (for each window, the excess over the allowed maximum).
        - objective_low_priority: similar violation count for low-priority options.
      The overall objective is then combined lexicographically according to the specified objective_order.
    """
    
    def __init__(self, instance_file=None, nb_positions=None, nb_options=None,
                 paint_batch_limit=None, objective_order=None, start_position=None,
                 max_cars_per_window=None, window_size=None, is_priority_option=None,
                 has_low_priority_options=None, color_class=None, options_data=None,
                 initial_sequence=None):
        if instance_file is not None:
            self._load_instance(instance_file)
        else:
            if (nb_positions is None or nb_options is None or paint_batch_limit is None or
                objective_order is None or start_position is None or max_cars_per_window is None or
                window_size is None or is_priority_option is None or has_low_priority_options is None or
                color_class is None or options_data is None or initial_sequence is None):
                raise ValueError("Either instance_file or all parameters must be provided.")
            self.nb_positions = nb_positions
            self.nb_options = nb_options
            self.paint_batch_limit = paint_batch_limit
            self.objective_order = objective_order
            self.start_position = start_position
            self.max_cars_per_window = max_cars_per_window
            self.window_size = window_size
            self.is_priority_option = is_priority_option
            self.has_low_priority_options = has_low_priority_options
            self.color_class = color_class
            self.options_data = options_data
            self.initial_sequence = initial_sequence

    def _load_instance(self, filename):
        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)
        # Read instance data as integers (for booleans, 1 indicates True)
        with open(filename, 'r') as f:
            tokens = f.read().split()
        it = iter(tokens)
        self.nb_positions = int(next(it))
        self.nb_options = int(next(it))
        nb_classes = int(next(it))
        self.paint_batch_limit = int(next(it))
        self.objective_order = int(next(it))
        self.start_position = int(next(it))
        # For each option: max_cars_per_window, window_size, is_priority_option
        self.max_cars_per_window = []
        self.window_size = []
        self.is_priority_option = []
        has_low = False
        for o in range(self.nb_options):
            self.max_cars_per_window.append(int(next(it)))
            self.window_size.append(int(next(it)))
            prio = (int(next(it)) == 1)
            self.is_priority_option.append(prio)
            if not prio:
                has_low = True
        self.has_low_priority_options = has_low
        # If there are no low priority options, adjust objective_order:
        if not has_low:
            if self.objective_order == COLOR_HIGH_LOW:
                self.objective_order = COLOR_HIGH
            elif self.objective_order == HIGH_COLOR_LOW:
                self.objective_order = HIGH_COLOR
            elif self.objective_order == HIGH_LOW_COLOR:
                self.objective_order = HIGH_COLOR
        # For each class: read color, number of cars, then for each option a binary indicator.
        self.color_class = []
        nb_cars_per_class = []
        self.options_data = []
        for c in range(nb_classes):
            self.color_class.append(int(next(it)))
            count = int(next(it))
            nb_cars_per_class.append(count)
            opts = [ (int(next(it)) == 1) for _ in range(self.nb_options) ]
            self.options_data.append(opts)
        # Build initial_sequence by repeating each class index the given number of times.
        self.initial_sequence = []
        for c in range(nb_classes):
            self.initial_sequence.extend([c] * nb_cars_per_class[c])
        if len(self.initial_sequence) != self.nb_positions:
            raise ValueError("Sum of cars per class does not equal nb_positions.")

    def evaluate_solution(self, solution) -> float:
        """
        Evaluate a candidate solution.

        The candidate solution should be a permutation (list) of integers of length nb_positions.
        It represents indices into the initial production plan.

        First, verify that positions before start_position remain fixed (i.e. candidate[p] == p for p < start_position).

        Then, reconstruct the production sequence:
           sequence[p] = initial_sequence[ solution[p] ]

        Compute three objectives:
          - objective_color: Sum over positions p = start_position-1 to nb_positions-2 of a 1 if
            the color (from color_class) changes between consecutive positions.
          - For each option o:
              For every window of length window_size[o] (from j = max(0, start_position - window_size[o] + 1)
              to nb_positions - window_size[o] + 1), count the number of cars (according to their class's requirement in options_data)
              that require option o. If the count exceeds max_cars_per_window[o], add the excess as a violation.
            Sum violations over all windows for high-priority options to get objective_high_priority, and similarly for low-priority options to get objective_low_priority.
          - Finally, combine the three objectives lexicographically according to objective_order.
            (We use a large constant M = 10000 to enforce lexicographic ordering.)

        Returns the overall objective value.
        """
        PENALTY = 1e9
        M = 10000
        if not isinstance(solution, (list, tuple)) or len(solution) != self.nb_positions:
            return PENALTY
        # Ensure the solution is a valid permutation.
        if sorted(solution) != list(range(self.nb_positions)):
            return PENALTY
        # Enforce fixed positions: for p in [0, start_position), candidate[p] must equal p.
        for p in range(self.start_position):
            if solution[p] != p:
                return PENALTY
        
        # Reconstruct production sequence: sequence[p] is the class index at position p.
        sequence = [self.initial_sequence[i] for i in solution]
        
        # Compute objective_color: count color changes from positions start_position-1 to nb_positions-2.
        objective_color = 0
        for p in range(max(self.start_position - 1, 0), self.nb_positions - 1):
            if self.color_class[ sequence[p] ] != self.color_class[ sequence[p+1] ]:
                objective_color += 1
        
        # Compute objective violations for options.
        objective_high_priority = 0
        objective_low_priority = 0
        # For each option o, evaluate over all valid windows.
        for o in range(self.nb_options):
            win = self.window_size[o]
            max_allowed = self.max_cars_per_window[o]
            # Window starting index: from max(0, start_position - win + 1) to nb_positions - win + 1.
            start_idx = max(0, self.start_position - win + 1)
            for j in range(start_idx, self.nb_positions - win + 1):
                count = 0
                for k in range(win):
                    car_class = sequence[j+k]
                    if self.options_data[car_class][o]:
                        count += 1
                violation = max(count - max_allowed, 0)
                if self.is_priority_option[o]:
                    objective_high_priority += violation
                else:
                    objective_low_priority += violation
        
        # Combine objectives lexicographically based on objective_order.
        if self.objective_order == COLOR_HIGH_LOW:
            overall = objective_color * (M**2) + objective_high_priority * M + objective_low_priority
        elif self.objective_order == HIGH_LOW_COLOR:
            overall = objective_high_priority * (M**2) + objective_low_priority * M + objective_color
        elif self.objective_order == HIGH_COLOR_LOW:
            overall = objective_high_priority * (M**2) + objective_color * M + objective_low_priority
        elif self.objective_order == COLOR_HIGH:
            overall = objective_color * M + objective_high_priority
        elif self.objective_order == HIGH_COLOR:
            overall = objective_high_priority * M + objective_color
        else:
            overall = objective_color + objective_high_priority + objective_low_priority
        
        return overall

    def random_solution(self):
        """
        Generate a random candidate solution.

        Returns a random permutation of 0,..., nb_positions-1 that respects the fixed positions:
        for positions p in [0, start_position), candidate[p] = p.
        """
        fixed = list(range(self.start_position))
        remaining = list(range(self.start_position, self.nb_positions))
        random.shuffle(remaining)
        return fixed + remaining
