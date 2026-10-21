"""Source-frame normal selection, restricted to pure visual methods."""
import numpy as np

GROUPS = ("F_RC", "F_R", "F_B", "V3_matched_R")


def constrained(normal, nominal):
    roll = np.arctan2(normal[0], -normal[1])
    pitch = np.arctan2(-nominal[2], np.linalg.norm(nominal[:2]))
    return np.array([np.cos(pitch) * np.sin(roll), -np.cos(pitch) * np.cos(roll), -np.sin(pitch)])


def pair_planes(group, nominal, records, output_index, height):
    if group not in GROUPS:
        raise ValueError(f"Unknown pure visual method: {group}")
    planes = []
    for source in range(output_index - 3, output_index):
        normal = np.asarray(nominal, float)
        if group in ("F_RC", "F_B"):
            record = records[source]
            if record["source_id"] != source:
                raise ValueError("Normal source-frame index mismatch")
            normal = np.asarray(record["normal"], float)
            if group == "F_RC":
                normal = constrained(normal, nominal)
        planes.append((normal, height))
    return planes
