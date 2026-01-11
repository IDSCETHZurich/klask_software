"""Game state machine definitions for Klask player."""

from enum import IntEnum, auto


class GameState(IntEnum):
    """States for the game state machine."""

    INITIALIZING = auto()      # Checking calibration status
    HOMING = auto()             # Waiting for homing to complete
    WAITING_FOR_READY = auto()  # Waiting for board to be ready
    PLAYING = auto()            # Normal gameplay - publishing actions
    GOAL_DETECTED = auto()      # Ball in goal - waiting for reset
    PEG_IN_GOAL = auto()        # Peg in goal - paused until clear
    RESETTING = auto()          # Calling homing service with delay


def state_name(state: GameState) -> str:
    """Get human-readable name for a state.
    
    Args:
        state: The game state
        
    Returns:
        String name of the state
    """
    return state.name
