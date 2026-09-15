"""Map-independent inertial racetrack with ordered, conservative grid collision.

State is (row, column, vx, vy); positive vy points UP. max_velocity is
exclusive (5 means speeds 0..4, or -4..4 in bidirectional mode).
"""
from ocean_rl.paths import MAPS_DIR
from ocean_rl.objectives import Transition, create_objective
from fractions import Fraction
import numpy as np
from ocean_rl.maps.io import MapSpec, from_ascii, from_image, load_map

# ---------------------------------------------------------------------------
# Graceful Fallback for Gymnasium (Farama Gymnasium / OpenAI Gym)
# ---------------------------------------------------------------------------
# If Gymnasium is installed, RacetrackEnv inherits from gym.Env to provide full
# compatibility with standard RL tools (e.g., Stable-Baselines3, Ray RLlib).
# If Gymnasium is absent, it degrades gracefully to a standard Python class.
try:
    import gymnasium as gym
    from gymnasium import spaces
    ENV_BASE = gym.Env
except ImportError:
    class ENV_BASE:
        pass
    spaces = None


def traversed_cells(row, col, delta_row, delta_col):
    """Yield simultaneous cell-entry groups along a center-to-center segment.

    Rational crossing times make corner ties exact. At a corner the two side
    cells and the diagonal cell are touched simultaneously (supercover).

    Parameters:
        row, col             : int - Starting grid coordinates
        delta_row, delta_col : int - Displacement vector (row delta, column delta)
    """
    row_step_sign, col_step_sign = int(np.sign(delta_row)), int(np.sign(delta_col))
    row_boundaries_crossed = 0
    col_boundaries_crossed = 0

    # Step through cell boundaries along the ray trajectory
    while row_boundaries_crossed < abs(delta_row) or col_boundaries_crossed < abs(delta_col):
        # Calculate exact fractional crossing time for the next row and column boundaries
        # Car starts at center of grid cell; to reach first boundary, it must travel 0.5 cell.
        next_row_cross_time = Fraction(2 * row_boundaries_crossed + 1, 2 * abs(delta_row)) if row_boundaries_crossed < abs(delta_row) else None
        next_col_cross_time = Fraction(2 * col_boundaries_crossed + 1, 2 * abs(delta_col)) if col_boundaries_crossed < abs(delta_col) else None

        if next_col_cross_time is None or (next_row_cross_time is not None and next_row_cross_time < next_col_cross_time):
            # Row boundary is crossed first -> move vertically. When moving vertical, the car cross each rows
            row += row_step_sign
            row_boundaries_crossed += 1
            yield [(row, col)]
        elif next_row_cross_time is None or next_col_cross_time < next_row_cross_time:
            # Column boundary is crossed first -> move horizontally
            col += col_step_sign
            col_boundaries_crossed += 1
            yield [(row, col)]
        else:
            # Exact corner tie (crossing row and column boundaries simultaneously). The middle of 4 cell intersection
            # Yield side cells and diagonal cell together to prevent corner-cutting bugs
            yield [(row + row_step_sign, col), (row, col + col_step_sign), (row + row_step_sign, col + col_step_sign)]
            row += row_step_sign
            col += col_step_sign
            row_boundaries_crossed += 1
            col_boundaries_crossed += 1


def make_action_grid(max_ax=1, max_ay=1):
    """Generate a discrete grid of (ax, ay) acceleration pairs.

    Parameters:
        max_ax : int - Maximum horizontal acceleration magnitude (creates range -max_ax..+max_ax)
        max_ay : int - Maximum vertical acceleration magnitude (creates range -max_ay..+max_ay)

    Returns:
        tuple of (ax, ay) 2-tuples
    """
    if not isinstance(max_ax, (int, np.integer)) or max_ax < 1:
        raise ValueError("max_ax must be a positive integer.")
    if not isinstance(max_ay, (int, np.integer)) or max_ay < 1:
        raise ValueError("max_ay must be a positive integer.")
    return tuple((ax, ay) for ax in range(-max_ax, max_ax + 1) for ay in range(-max_ay, max_ay + 1))


class RacetrackEnv(ENV_BASE):
    """Inertial Grid Racetrack Environment.

    The car operates on a 2D grid with discrete velocity and acceleration.
    - Grid values: 0 = Wall, 1 = Track, 2 = Start Line, 3 = Finish Line.
    - Actions: Configurable discrete accelerations (default: 9 accelerations ax, ay in {-1, 0, 1}).
    - State: (row, col, vx, vy).
    """

    # Default 9 Discrete acceleration choices: (-1,-1), (-1,0), (-1,1), ..., (1,1)
    ACTION_MAP = tuple((ax, ay) for ax in (-1, 0, 1) for ay in (-1, 0, 1))
    actions = ACTION_MAP
    action_count = len(actions)
    metadata = {'render_modes': []}

    def __init__(self, grid=None, track_id=1, fail_prob=0.10, max_velocity=5,
                 *, map_spec=None, movement_mode='forward-only', max_episode_steps=10000, seed=None,
                 objective='minimum-time', custom_actions=None):
        """Initialize Racetrack Environment.

        Parameters:
            grid              : 2D numpy array (optional if map_spec or track_id provided)
            track_id          : Standard benchmark map ID (1 = track_1.map)
            fail_prob         : Probability of acceleration failure (noise / engine slip)
            max_velocity      : Exclusive speed bound (e.g., 5 means speeds 0..4)
            movement_mode     : 'forward-only' (non-negative speeds) or 'bidirectional' (reverse allowed)
            max_episode_steps : Truncation threshold for long episodes
            seed              : PRNG seed for reproducibility
            objective         : Objective function name ('minimum-time', etc.)
            custom_actions    : Optional sequence of (ax, ay) 2-tuples for flexible action sets
        """
        super().__init__()
        self.objective = objective
        self.reward_objective = create_objective(objective)

        # Action set validation and setup
        if custom_actions is not None:
            if not isinstance(custom_actions, (list, tuple)) or len(custom_actions) == 0:
                raise ValueError("custom_actions must be a non-empty sequence of (ax, ay) 2-tuples.")
            validated = []
            for item in custom_actions:
                if not (isinstance(item, (tuple, list)) and len(item) == 2 and
                        not isinstance(item[0], bool) and isinstance(item[0], (int, np.integer)) and
                        not isinstance(item[1], bool) and isinstance(item[1], (int, np.integer))):
                    raise ValueError("Each action in custom_actions must be a 2-tuple of integers (ax, ay).")
                validated.append((int(item[0]), int(item[1])))
            self.actions = tuple(validated)
        else:
            self.actions = self.ACTION_MAP
        self.action_count = len(self.actions)

        # Parameter validation
        if isinstance(max_velocity, bool) or not isinstance(max_velocity, (int, np.integer)) or max_velocity < 2:
            raise ValueError('max_velocity must be an integer >= 2 (exclusive speed bound).')
        if not np.isfinite(fail_prob) or not 0 <= fail_prob <= 1:
            raise ValueError('fail_prob must be finite in [0, 1].')
        if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, (int, np.integer)) or max_episode_steps < 1:
            raise ValueError('max_episode_steps must be a positive integer.')
        if grid is not None and map_spec is not None:
            raise ValueError('Provide grid or map_spec, not both.')

        # ---------------------------------------------------------------------------
        # Map Resolution & Solvability Validation
        # ---------------------------------------------------------------------------
        # Map source priority:
        # 1. Pre-constructed map_spec object (passed directly or via from_file/from_image/from_ascii)
        # 2. Raw 2D NumPy grid array (wrapped into a MapSpec instance)
        # 3. Default benchmark map file (track_1.map from MAPS_DIR)
        if map_spec is None:
            if grid is not None:
                map_spec = MapSpec(grid)
            else:
                if track_id != 1:
                    raise ValueError('Only track_id=1 is registered; use from_file for other maps.')
                map_spec = load_map(MAPS_DIR / 'track_1.map')
        if not isinstance(map_spec, MapSpec):
            raise TypeError('map_spec must be a MapSpec.')

        # Run Breadth-First Search (BFS) pathfinding to verify every start line cell (2)
        # can reach at least one finish line cell (3) under the current movement mode.
        map_spec.validate_reachability(movement_mode)

        # Map grid properties and cell label mappings
        self.map_spec = map_spec
        self.grid = map_spec.grid                           # 2D grid matrix: 0=Wall, 1=Track, 2=Start, 3=Finish
        self.height, self.width = self.grid.shape          # Grid row (height) and column (width) dimensions
        self.start_cells = map_spec.start_cells            # List of (row, col) coordinates for Start Line cells (2)
        self.finish_cells = map_spec.finish_cells          # List of (row, col) coordinates for Finish Line cells (3)
        self.fail_prob, self.max_velocity = float(fail_prob), int(max_velocity)
        self.movement_mode, self.max_episode_steps = movement_mode, int(max_episode_steps)
        self.track_id = track_id

        # Determine velocity bounds based on movement mode
        self.velocity_min = 0 if movement_mode == 'forward-only' else 1 - self.max_velocity #Reverse allowed in bidirectional
        self.velocity_max = self.max_velocity - 1

        self.rng = np.random.default_rng(seed)
        self.current_state = None
        self.elapsed_steps = 0
        self._done = False

        # Set up Gymnasium spaces if available
        if spaces is not None:
            self.action_space = spaces.Discrete(self.action_count, seed=seed)
            self.observation_space = spaces.Box(
                low=np.array([0, 0, self.velocity_min, self.velocity_min], dtype=np.int32),
                high=np.array([self.height - 1, self.width - 1, self.velocity_max, self.velocity_max], dtype=np.int32),
                dtype=np.int32)

    # ---------------------------------------------------------------------------
    # Factory Constructors & Cloning
    # ---------------------------------------------------------------------------
    @classmethod
    def from_file(cls, path, **overrides):
        """Construct environment from a file path (.map, .json, .tmj)."""
        spec = load_map(path)
        options = dict(spec.metadata)
        allowed = {'fail_prob', 'max_velocity', 'movement_mode', 'max_episode_steps', 'seed', 'objective', 'custom_actions'}
        if set(options) - allowed:
            raise ValueError(f'Unknown environment configuration keys: {sorted(set(options) - allowed)}')
        options.update(overrides)
        return cls(map_spec=spec, **options)

    @classmethod
    def from_numpy(cls, grid_array, fail_prob=0.10, max_velocity=5, **kwargs):
        """Construct environment directly from a 2D NumPy array grid."""
        return cls(grid=grid_array, fail_prob=fail_prob, max_velocity=max_velocity, **kwargs)

    @classmethod
    def from_ascii(cls, ascii_source, fail_prob=0.10, max_velocity=5, char_map=None, **kwargs):
        """Construct environment from ASCII text representation."""
        return cls(map_spec=from_ascii(ascii_source, char_map), fail_prob=fail_prob, max_velocity=max_velocity, **kwargs)

    @classmethod
    def from_image(cls, image_source, fail_prob=0.10, max_velocity=5, wall_thresh=50,
                   *, palette=None, start_cells=None, finish_cells=None, **kwargs):
        """Construct environment from an image file (PNG/BMP)."""
        spec = from_image(image_source, palette=palette, start_cells=start_cells,
                          finish_cells=finish_cells, wall_thresh=wall_thresh)
        return cls(map_spec=spec, fail_prob=fail_prob, max_velocity=max_velocity, **kwargs)

    def clone(self, **overrides):
        """Create a new instance with identical map and settings."""
        options = dict(fail_prob=self.fail_prob, max_velocity=self.max_velocity,
                       movement_mode=self.movement_mode, max_episode_steps=self.max_episode_steps,
                       custom_actions=self.actions)
        options['objective'] = self.objective
        options.update(overrides)
        return type(self)(map_spec=self.map_spec, **options)

    # ---------------------------------------------------------------------------
    # Environment Lifecycle & State Management
    # ---------------------------------------------------------------------------
    def _random_start(self):
        """Select a random starting cell coordinate with zero velocity."""
        start_row, start_col = self.start_cells[self.rng.integers(len(self.start_cells))]
        return (int(start_row), int(start_col), 0, 0)

    def reset(self, seed=None, options=None):
        """Reset environment state to start a new episode.

        Returns:
            observation : np.ndarray - Initial state [row, col, vx, vy]
            info        : dict       - Additional metadata
        """
        if spaces is not None:
            super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if spaces is not None:
                self.action_space.seed(seed)
        options = {} if options is None else options
        if set(options) - {'start_cell'}:
            raise ValueError('The only reset option is start_cell.')
        if 'start_cell' in options:
            cell = np.asarray(options['start_cell'])
            if cell.shape != (2,) or cell.dtype.kind not in 'iu' or not any(np.array_equal(cell, start) for start in self.start_cells):
                raise ValueError('start_cell must be an integer coordinate labeled as a start.')
            self.current_state = (*map(int, cell), 0, 0)
        else:
            self.current_state = self._random_start()
        self.elapsed_steps, self._done = 0, False
        return np.array(self.current_state, dtype=np.int32), {}

    def get_valid_actions(self, state):
        """Return list of valid action indices from the current state.

        Valid actions must keep velocity within bounds and prevent complete
        stopping (vx=0, vy=0) unless currently located on a start cell.
        """
        current_row, current_col, vel_x, vel_y = map(int, state)
        return [index for index, (accel_x, accel_y) in enumerate(self.actions)
                if self.velocity_min <= vel_x + accel_x <= self.velocity_max
                and self.velocity_min <= vel_y + accel_y <= self.velocity_max
                and ((vel_x + accel_x, vel_y + accel_y) != (0, 0) or
                     (self.grid[current_row, current_col] == 2 and self.reward_objective.allow_stationary))]

    def step(self, action):
        """Execute one environment step given an action index.

        Parameters:
            action : int - Action index (0..8)

        Returns:
            obs        : np.ndarray - New state [row, col, nvx, nvy]
            reward     : float      - Step reward from objective function
            terminated : bool       - True if finish line reached
            truncated  : bool       - True if max_episode_steps exceeded
            info       : dict       - Metadata (crashed, distance, etc.)
        """
        if self.current_state is None or self._done:
            raise RuntimeError('Call reset before stepping a new episode.')
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < self.action_count:
            raise ValueError('action must be an integer in the discrete action space.')
        current_row, current_col, vel_x, vel_y = self.current_state
        if not self.reward_objective.allow_stationary and action not in self.get_valid_actions(self.current_state):
            raise ValueError('Action violates the objective movement mask')

        # Roll stochastic engine failure (fail_prob chance of zero acceleration)
        failed = self.rng.random() < self.fail_prob
        accel_x, accel_y = (0, 0) if failed else self.actions[action]

        # Update and clip velocity
        next_vel_x = int(np.clip(vel_x + accel_x, self.velocity_min, self.velocity_max))
        next_vel_y = int(np.clip(vel_y + accel_y, self.velocity_min, self.velocity_max))

        # Prevent non-start stopping: retain previous velocity if updated velocity is zero off start line
        if next_vel_x == next_vel_y == 0 and self.grid[current_row, current_col] != 2:
            next_vel_x, next_vel_y = vel_x, vel_y

        crashed = terminated = False
        target = (current_row - next_vel_y, current_col + next_vel_x)  # Grid row decreases going UP (-next_vel_y)
        distance = float(np.hypot(next_vel_x, next_vel_y))

        def entry_fraction(cell):
            """Calculate fractional ray contact distance to cell boundary."""
            entries = [0.0]
            for origin, delta, coordinate in zip((current_row, current_col), (-next_vel_y, next_vel_x), cell):
                if delta:
                    entries.append(min((coordinate - .5 - origin) / delta,
                                       (coordinate + .5 - origin) / delta))
            return max(entries)

        # Ray-march along the traversed cells between current position and target position
        for cells in traversed_cells(current_row, current_col, -next_vel_y, next_vel_x):
            # Check for wall collision (grid == 0 or out of bounds)
            if any(not (0 <= cell_row < self.height and 0 <= cell_col < self.width) or self.grid[cell_row, cell_col] == 0 for cell_row, cell_col in cells):
                crashed = True
                distance *= min(entry_fraction(cell) for cell in cells)
                break
            # Check for finish line crossing (grid == 3)
            finishes = [cell for cell in cells if self.grid[cell] == 3]
            if finishes:
                target = finishes[0] #Clamps the car's final position to the finish line cell
                terminated = True
                distance *= min(entry_fraction(cell) for cell in finishes)
                break

        # State transition: Reset to start cell on crash; otherwise update position and velocity
        self.current_state = self._random_start() if crashed else (*target, next_vel_x, next_vel_y)
        self.elapsed_steps += 1
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        self._done = terminated or truncated

        # Compute step reward using the selected objective
        reward = float(self.reward_objective.reward(Transition(distance, terminated, crashed)))
        if not np.isfinite(reward):
            raise ValueError('Objective returned a nonfinite reward')

        return (np.array(self.current_state, dtype=np.int32), reward,
                terminated, truncated, {'crashed': crashed, 'acceleration_failed': failed,
                                        'distance': distance, 'objective': self.objective})
