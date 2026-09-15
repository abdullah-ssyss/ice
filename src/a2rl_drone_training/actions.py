"""Versioned policy action semantics shared by training and evaluation."""

ACTION_SPACE = "motor_rpm_hover_centered_v1"
ACTION_NAMES = ("motor_1", "motor_2", "motor_3", "motor_4")


def validate_checkpoint_actions(payload: dict) -> None:
    if payload.get("action_space") != ACTION_SPACE:
        raise ValueError(
            "Checkpoint action space is incompatible with direct motor control. "
            "Attitude-control checkpoints cannot be resumed or evaluated as motor "
            "policies. Start a fresh run in a new checkpoint directory."
        )
