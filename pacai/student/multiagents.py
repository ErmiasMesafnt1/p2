
import typing

import pacai.agents.greedy
import pacai.agents.minimax
import pacai.core.action
import pacai.core.board
import pacai.core.gamestate
import pacai.pacman.board
import pacai.pacman.gamestate


def _manhattan_distance(a: pacai.core.board.Position, b: pacai.core.board.Position) -> int:
    """Return the Manhattan distance between two points without pulling in extra helpers."""

    return abs(a.row - b.row) + abs(a.col - b.col)


def _closest_distance(origin: pacai.core.board.Position,
        targets: typing.Iterable[pacai.core.board.Position]) -> int | None:
    """Return the minimum distance from *origin* to the iterable of *targets* (if any)."""

    distances = [_manhattan_distance(origin, target) for target in targets]
    if (len(distances) == 0):
        return None

    return min(distances)

class ReflexAgent(pacai.agents.greedy.GreedyAgent):
    """
    A simple agent based on pacai.agents.greedy.GreedyAgent.

    You job is to make this agent better (it is pretty bad right now).
    You can change whatever you want about it,
    but it should still be a child of pacai.agents.greedy.GreedyAgent
    and be a "reflex" agent.
    This means that it shouldn't do any formal planning or searching,
    instead it should just look at the state of the game and try to make a good choice in the moment.
    You can make a great agent just by implementing a custom evaluate_state() method
    (and maybe add to the constructor if you want).
    """

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        # Put code here if you want.

    def evaluate_state(self,
            state: pacai.core.gamestate.GameState,
            action: pacai.core.action.Action | None = None,
            **kwargs: typing.Any) -> float:
        if (state.game_over):
            winners = state.game_complete()
            return float('inf') if (
                pacai.pacman.gamestate.PACMAN_AGENT_INDEX in winners) else float('-inf')

        pacman_position = state.get_agent_position(pacai.pacman.gamestate.PACMAN_AGENT_INDEX)
        if (pacman_position is None):
            return float('-inf')

        score = float(state.score)

        food_positions = [food for food in state.get_food()]
        closest_food = _closest_distance(pacman_position, food_positions)
        if (closest_food is not None):
            score += 9.0 / (closest_food + 1.0)
            score -= 0.6 * len(food_positions)

        capsules = list(state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE))
        closest_capsule = _closest_distance(pacman_position, capsules)
        if (closest_capsule is not None):
            score += 4.5 / (closest_capsule + 1.0)
            score -= 1.0 * len(capsules)

        ghosts = _closest_distance(
            pacman_position,
            (position for position in state.get_nonscared_ghost_positions().values()
                if position is not None))
        if (ghosts is not None):
            if (ghosts == 0):
                return float('-inf')

            score -= 7.0 / ghosts
            if (ghosts <= 2):
                score -= 12.0 / ghosts

        scared_ghosts = _closest_distance(
            pacman_position,
            (position for position in state.get_scared_ghost_positions().values()
                if position is not None))
        if (scared_ghosts is not None):
            score += 5.0 / (scared_ghosts + 1.0)

        if (len(self.last_positions) >= 2):
            previous_position = self.last_positions[-2]
            if (previous_position is not None and previous_position == pacman_position):
                score -= 2.0

        return score

class MyMinimaxLikeAgent(pacai.agents.minimax.MinimaxLikeAgent):
    """
    An agent that implements all the required methods for the minimax family of algorithms.
    Default implementations are supplied, so the agent should run right away,
    but it will not be very good.

    To implement minimax, minimax_step_max() and minimax_step_min() are required
    (you can ignore alpha and beta).

    To implement minimax with alpha-beta pruning,
    minimax_step_max() and minimax_step_min() with alpha and beta are required.

    To implement expectimax, minimax_step_max() and minimax_step_expected_min() are required.

    You are free to implement/override any methods you need to.
    """

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        # You can use the constructor if you need to.

    def minimax_step_max(self,
            state: pacai.core.gamestate.GameState,
            ply_count: int,
            legal_actions: list[pacai.core.action.Action],
            alpha: float,
            beta: float,
            ) -> tuple[list[pacai.core.action.Action], float]:
        if (len(legal_actions) == 0):
            return [], self.evaluate_state(state)

        best_score = float('-inf')
        best_actions: list[pacai.core.action.Action] = []

        for action in legal_actions:
            successor = state.generate_successor(action, self.rng)
            _, score = self.minimax_step(successor, ply_count, alpha, beta)

            if (score > best_score):
                best_score = score
                best_actions = [action]
            elif (score == best_score):
                best_actions.append(action)

            if (self.alphabeta_prune):
                alpha = max(alpha, best_score)
                if (alpha >= beta or best_score == float('inf')):
                    break

        return best_actions, best_score

    def minimax_step_min(self,
            state: pacai.core.gamestate.GameState,
            ply_count: int,
            legal_actions: list[pacai.core.action.Action],
            alpha: float,
            beta: float,
            ) -> tuple[list[pacai.core.action.Action], float]:
        if (len(legal_actions) == 0):
            return [], self.evaluate_state(state)

        best_score = float('inf')
        best_actions: list[pacai.core.action.Action] = []

        for action in legal_actions:
            successor = state.generate_successor(action, self.rng)
            _, score = self.minimax_step(successor, ply_count, alpha, beta)

            if (score < best_score):
                best_score = score
                best_actions = [action]
            elif (score == best_score):
                best_actions.append(action)

            if (self.alphabeta_prune):
                beta = min(beta, best_score)
                if (beta <= alpha or best_score == float('-inf')):
                    break

        return best_actions, best_score

    def minimax_step_expected_min(self,
            state: pacai.core.gamestate.GameState,
            ply_count: int,
            legal_actions: list[pacai.core.action.Action],
            alpha: float,
            beta: float,
            ) -> float:
        if (len(legal_actions) == 0):
            return self.evaluate_state(state)

        probability = 1.0 / len(legal_actions)
        expectation = 0.0

        for action in legal_actions:
            successor = state.generate_successor(action, self.rng)
            _, score = self.minimax_step(successor, ply_count, alpha, beta)

            if (score in (float('inf'), float('-inf'))):
                return score

            expectation += probability * score

        return expectation

def better_state_eval(
        state: pacai.core.gamestate.GameState,
        agent: typing.Any | None = None,
        action: pacai.core.action.Action | None = None,
        **kwargs: typing.Any) -> float:
    """
    Create a better state evaluation function for your MyMinimaxLikeAgent agent!
    
    Combines the game score with simple heuristics:
    - Rewards getting closer to food and capsules.
    - Penalizes being near active ghosts.
    - Encourages chasing ghosts when they’re scared.
    Uses inverse distances to nearby targets and scales penalties
    based on how close the danger is.
    """
   
    
    if (state.game_over):
        winners = state.game_complete()
        return float('inf') if (
            pacai.pacman.gamestate.PACMAN_AGENT_INDEX in winners) else float('-inf')

    pacman_position = state.get_agent_position(pacai.pacman.gamestate.PACMAN_AGENT_INDEX)
    if (pacman_position is None):
        return float('-inf')

    score = float(state.score)

    food = list(state.get_food())
    if (len(food) > 0):
        nearest_three = sorted(
            (_manhattan_distance(pacman_position, pellet) for pellet in food))[:3]
        score += sum(12.0 / (distance + 1.0) for distance in nearest_three)
        score -= 3.5 * len(food)

    capsules = list(state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE))
    closest_capsule = _closest_distance(pacman_position, capsules)
    if (closest_capsule is not None):
        score += 18.0 / (closest_capsule + 1.0)
        score -= 8.0 * len(capsules)

    nonscared = [
        position for position in state.get_nonscared_ghost_positions().values()
        if position is not None
    ]
    closest_hostile = _closest_distance(pacman_position, nonscared)
    if (closest_hostile is not None):
        if (closest_hostile == 0):
            return float('-inf')

        score -= 40.0 / closest_hostile
        if (closest_hostile <= 2):
            score -= 80.0 / closest_hostile

    scared = {
        index: position for (index, position) in state.get_scared_ghost_positions().items()
        if position is not None
    }
    for (index, position) in scared.items():
        distance = _manhattan_distance(pacman_position, position)
        timer = getattr(state, 'scared_timers', {}).get(index, 0)
        timer_bonus = 1.0 + (timer / pacai.pacman.gamestate.SCARED_TIME)
        score += timer_bonus * (35.0 / (distance + 1.0))

    return score
