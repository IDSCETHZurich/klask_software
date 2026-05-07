"""Shared debug utilities — visualize image tensors and ego-frame state with OpenCV."""

import cv2
import numpy as np
import torch


def show_image_tensor(tensor: torch.Tensor, window_name: str = "dreamer_debug") -> None:
    """Show an image tensor in an OpenCV window.

    Accepts shape (1, H, W, 3) or (H, W, 3), RGB, float in [0, 255] or [0, 1],
    or uint8. Batch dim is dropped; floats are rescaled and clipped; RGB is
    converted to BGR. Uses a non-blocking waitKey(1) so this can be called
    from the inference loop.
    """
    img = tensor.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    img = img.numpy()

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, img_bgr)
    cv2.waitKey(1)


# --- State observation renderer -------------------------------------------------

# Canvas layout (px). Forward axis (ego y) is drawn vertical, +forward up.
_BOARD_PX = 360            # forward-axis size of the board area
_LATERAL_PX = 280          # lateral-axis size of the board area
_PANEL_PX = 270            # text panel width (left side of canvas)
_MARGIN_PX = 20            # margin around board
_LINE_H = 18               # text panel line height

# Colors (BGR)
_C_BG = (30, 30, 30)
_C_BOARD = (200, 200, 200)
_C_GOAL_OWN = (180, 180, 80)
_C_GOAL_OPP = (80, 180, 180)
_C_PLAYER = (255, 120, 60)        # blue-ish
_C_OPPONENT = (60, 60, 220)       # red
_C_BALL = (60, 220, 220)          # yellow
_C_VEL = (255, 255, 255)
_C_LINE_PB = (180, 220, 120)      # player→ball
_C_LINE_POG = (120, 220, 180)     # player→opp_goal
_C_LINE_OB = (180, 120, 220)      # opp→ball
_C_LINE_OOG = (220, 180, 120)     # opp→own_goal
_C_TEXT = (230, 230, 230)


def _ego_to_px(p_ego, origin_px, px_per_m):
    """Map an ego-frame point (meters) to pixel (col, row).

    Layout follows map_state_observations(): p_ego[0] is the lateral axis
    (msg.y, span = board_dim_height) and p_ego[1] is the forward / goal-to-goal
    axis (msg.x, span = board_dim_width). +forward is drawn upward
    (decreasing row); +lateral is drawn rightward.
    """
    lateral, forward = float(p_ego[0]), float(p_ego[1])
    col = int(round(origin_px[0] + lateral * px_per_m))
    row = int(round(origin_px[1] - forward * px_per_m))
    return (col, row)


def show_state_observation(
    obs_base: np.ndarray,
    obs_full: np.ndarray,
    board_dim_width: float,
    board_dim_height: float,
    window_name: str = "state_debug",
) -> None:
    """Render ego-frame state observation in an OpenCV window.

    Args:
        obs_base: 16-D vector from map_state_observations() — layout
            [player_pos(2), player_vel(2), opponent_pos(2), opponent_vel(2),
             ball_pos(2), ball_vel(2), goal_player_pos(2), goal_opponent_pos(2)].
            Each pair is [forward_y, lateral_x] in meters.
        obs_full: 20-D vector from add_additional_state_features() — first 12
            entries match obs_base[:12]; last 8 are
            [angle_pegball_pegoppgoal, angle_oppball_oppgoal,
             angle_pegball_pegopp, angle_oppball_pegopp,
             distance_ball_goal, distance_ball_oppgoal,
             distance_ball_player, distance_ball_opp].
        board_dim_width: long board axis (forward direction), meters.
        board_dim_height: short board axis (lateral direction), meters.
        window_name: OpenCV window name.
    """
    # Pixels-per-meter chosen so the forward axis fits _BOARD_PX.
    px_per_m = _BOARD_PX / board_dim_width

    board_w_px = int(round(board_dim_height * px_per_m))   # lateral
    board_h_px = int(round(board_dim_width * px_per_m))    # forward

    # Layout: text panel on the LEFT, board on the RIGHT.
    panel_x = _MARGIN_PX
    board_left = _MARGIN_PX + _PANEL_PX + _MARGIN_PX
    board_top = _MARGIN_PX

    canvas_w = board_left + board_w_px + _MARGIN_PX
    # Canvas height grows to fit either the board or the text panel,
    # whichever is taller. Panel: ~24 rows (positions, velocities, goals,
    # angles, distances + headers + spacers).
    panel_rows = 24
    panel_h = _MARGIN_PX + panel_rows * _LINE_H + _MARGIN_PX
    canvas_h = max(_MARGIN_PX + board_h_px + _MARGIN_PX, panel_h)
    canvas = np.full((canvas_h, canvas_w, 3), _C_BG, dtype=np.uint8)

    # Origin (ego 0,0) is the center of the board area.
    origin_col = board_left + board_w_px // 2
    origin_row = board_top + board_h_px // 2
    origin_px = (origin_col, origin_row)

    # Board outline.
    cv2.rectangle(
        canvas,
        (board_left, board_top),
        (board_left + board_w_px, board_top + board_h_px),
        _C_BOARD,
        1,
    )

    # Unpack ego-frame entities from obs_base.
    player_pos = obs_base[0:2]
    player_vel = obs_base[2:4]
    opp_pos = obs_base[4:6]
    opp_vel = obs_base[6:8]
    ball_pos = obs_base[8:10]
    ball_vel = obs_base[10:12]
    goal_own = obs_base[12:14]
    goal_opp = obs_base[14:16]

    # Goals — small filled circles at goal centers. In ego frame the player's
    # own goal sits on the -forward side and opponent's goal on +forward side.
    cv2.circle(canvas, _ego_to_px(goal_own, origin_px, px_per_m), 8, _C_GOAL_OWN, -1)
    cv2.circle(canvas, _ego_to_px(goal_opp, origin_px, px_per_m), 8, _C_GOAL_OPP, -1)

    # Angle-defining lines (drawn before pegs/ball so circles stay on top).
    p_px = _ego_to_px(player_pos, origin_px, px_per_m)
    o_px = _ego_to_px(opp_pos, origin_px, px_per_m)
    b_px = _ego_to_px(ball_pos, origin_px, px_per_m)
    gown_px = _ego_to_px(goal_own, origin_px, px_per_m)
    gopp_px = _ego_to_px(goal_opp, origin_px, px_per_m)
    cv2.line(canvas, p_px, b_px, _C_LINE_PB, 1)
    cv2.line(canvas, p_px, gopp_px, _C_LINE_POG, 1)
    cv2.line(canvas, o_px, b_px, _C_LINE_OB, 1)
    cv2.line(canvas, o_px, gown_px, _C_LINE_OOG, 1)

    # Velocity arrows. Scale velocity vectors so 1 m/s ≈ 0.1 m of board (visual).
    vel_scale = 0.1
    def _vel_arrow(pos, vel, color):
        tip = np.array([pos[0] + vel[0] * vel_scale, pos[1] + vel[1] * vel_scale])
        cv2.arrowedLine(
            canvas,
            _ego_to_px(pos, origin_px, px_per_m),
            _ego_to_px(tip, origin_px, px_per_m),
            color,
            1,
            tipLength=0.3,
        )

    _vel_arrow(player_pos, player_vel, _C_VEL)
    _vel_arrow(opp_pos, opp_vel, _C_VEL)
    _vel_arrow(ball_pos, ball_vel, _C_VEL)

    # Entities (player blue, opponent red, ball yellow).
    cv2.circle(canvas, p_px, 7, _C_PLAYER, -1)
    cv2.circle(canvas, o_px, 7, _C_OPPONENT, -1)
    cv2.circle(canvas, b_px, 5, _C_BALL, -1)

    # Text panel (LEFT side) — positions, velocities, goals, angles, distances.
    angles_rad = obs_full[12:16]
    distances = obs_full[16:20]
    angles_deg = np.degrees(angles_rad)

    y = _MARGIN_PX + _LINE_H
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45

    def _txt(text, color=_C_TEXT):
        nonlocal y
        cv2.putText(canvas, text, (panel_x, y), font, fs, color, 1, cv2.LINE_AA)
        y += _LINE_H

    def _vec(p):
        """Format an ego (lateral, forward) pair as '(+0.040, -0.100)'."""
        return f"({p[0]:+.3f}, {p[1]:+.3f})"

    _txt("positions [m] (lat, fwd)")
    _txt(f"  player  : {_vec(player_pos)}", _C_PLAYER)
    _txt(f"  opp     : {_vec(opp_pos)}", _C_OPPONENT)
    _txt(f"  ball    : {_vec(ball_pos)}", _C_BALL)
    _txt(f"  own_g   : {_vec(goal_own)}", _C_GOAL_OWN)
    _txt(f"  opp_g   : {_vec(goal_opp)}", _C_GOAL_OPP)
    y += _LINE_H // 2

    _txt("velocities [m/s] (lat, fwd)")
    _txt(f"  player  : {_vec(player_vel)}", _C_PLAYER)
    _txt(f"  opp     : {_vec(opp_vel)}", _C_OPPONENT)
    _txt(f"  ball    : {_vec(ball_vel)}", _C_BALL)
    y += _LINE_H // 2

    _txt("angles [deg]")
    _txt(f"  pb-pog : {angles_deg[0]:6.1f}", _C_LINE_PB)
    _txt(f"  ob-og  : {angles_deg[1]:6.1f}", _C_LINE_OB)
    _txt(f"  pb-po  : {angles_deg[2]:6.1f}")
    _txt(f"  ob-po  : {angles_deg[3]:6.1f}")
    y += _LINE_H // 2
    _txt("distances [m]")
    _txt(f"  ball-own_g : {distances[0]:.3f}")
    _txt(f"  ball-opp_g : {distances[1]:.3f}")
    _txt(f"  ball-player: {distances[2]:.3f}")
    _txt(f"  ball-opp   : {distances[3]:.3f}")

    cv2.imshow(window_name, canvas)
    cv2.waitKey(1)
