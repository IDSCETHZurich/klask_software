"""Game state machine definitions for Klask player."""

from enum import IntEnum, auto


class GameState(IntEnum):
    """States for the game state machine."""

    INITIALIZING = auto()
    STATE_ESTIMATOR_READY = auto()
    HW_CALIBRATED = auto()
    HW_UNCALIBRATED = auto()
    HW_READY = auto()
    REQUESTING_HOME_CAL = auto()
    HOMING = auto()
    INTERACTION_DELAY = auto()
    PLAYING = auto()
    GAME_OVER = auto()
    MOVE_MAGNET = auto()
    WAIT_FOR_PEG_RESET = auto()
    UNKNOWN_BOARD_STATE = auto()


def state_name(state: GameState) -> str:
    """Get human-readable name for a state.

    Args:
        state: The game state

    Returns:
        String name of the state
    """
    return state.name
