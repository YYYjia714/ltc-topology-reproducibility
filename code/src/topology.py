from __future__ import annotations

JOINT_NAMES_25 = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
    "jaw",
]

SKELETON_EDGES_25 = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 4],
    [2, 5],
    [3, 6],
    [4, 7],
    [5, 8],
    [6, 9],
    [7, 10],
    [8, 11],
    [9, 12],
    [12, 13],
    [12, 14],
    [12, 15],
    [13, 16],
    [14, 17],
    [16, 18],
    [17, 19],
    [18, 20],
    [19, 21],
    [20, 22],
    [21, 23],
    [15, 24],
]


def get_joint_names(num_joints: int) -> list[str]:
    if num_joints <= len(JOINT_NAMES_25):
        return JOINT_NAMES_25[:num_joints]
    return [f"joint_{index}" for index in range(num_joints)]


def get_edges(num_joints: int) -> list[list[int]]:
    return [edge for edge in SKELETON_EDGES_25 if edge[0] < num_joints and edge[1] < num_joints]
