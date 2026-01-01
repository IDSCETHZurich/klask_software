from enum import IntFlag


class BoardState(IntFlag):
    UNKNOWN = 0
    READY = 1 << 0
    BALL_IN_LEFT_GOAL = 1 << 1
    BALL_IN_RIGHT_GOAL = 1 << 2
    PEG_IN_LEFT_GOAL = 1 << 3
    PEG_IN_RIGHT_GOAL = 1 << 4
