import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

# =========================
# CONFIGURACIÓN
# =========================

SEED = 42
OUTPUT_DIR = "."

ZONES = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
OBJECTS = ["starBlackPos", "starOrangePos", "cubeBlackPos", "cubeOrangePos"]
ROTATIONS = ["low", "mid", "high"]

CLUSTER_PENALTY_WEIGHT = 0.35
CANDIDATES_PER_EPISODE = 500


# =========================
# FUNCIONES AUXILIARES
# =========================

def zone_to_coord(zone):
    return {"A": 0, "B": 1, "C": 2}[zone[0]], int(zone[1]) - 1


def average_pairwise_distance(zones):
    coords = [zone_to_coord(z) for z in zones]
    distances = [
        abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
        for i in range(len(coords))
        for j in range(i + 1, len(coords))
    ]
    return sum(distances) / len(distances)


def generate_candidate():
    zones = random.sample(ZONES, len(OBJECTS))
    return {
        "positions": dict(zip(OBJECTS, zones)),
        "rot": random.choice(ROTATIONS),
    }


def score_candidate(candidate, obj_zone_counts, global_zone_counts, rot_counts):
    score = 0.0
    positions = candidate["positions"]

    for obj, zone in positions.items():
        score += obj_zone_counts[obj][zone] * 2.0
        score += global_zone_counts[zone] * 0.8

    score += rot_counts[candidate["rot"]] * 1.5

    avg_dist = average_pairwise_distance(list(positions.values()))
    score += max((4.0 - avg_dist) * CLUSTER_PENALTY_WEIGHT, 0)

    return score


def select_best_candidate(obj_zone_counts, global_zone_counts, rot_counts):
    best, best_score = None, float("inf")
    for _ in range(CANDIDATES_PER_EPISODE):
        c = generate_candidate()
        s = score_candidate(c, obj_zone_counts, global_zone_counts, rot_counts)
        if s < best_score:
            best_score, best = s, c
    return best


def generate_layouts(start_id, end_id):
    """Genera episodios con IDs desde start_id hasta end_id (inclusive)."""
    obj_zone_counts = defaultdict(Counter)
    global_zone_counts = Counter()
    rot_counts = Counter()
    layouts = []

    for episode_id in range(start_id, end_id + 1):
        c = select_best_candidate(obj_zone_counts, global_zone_counts, rot_counts)

        for obj, zone in c["positions"].items():
            obj_zone_counts[obj][zone] += 1
            global_zone_counts[zone] += 1
        rot_counts[c["rot"]] += 1

        layouts.append({
            "id": episode_id,
            **c["positions"],
            "rot": c["rot"],
        })

    return layouts


def save_to_csv(layouts, output_file):
    fieldnames = ["id"] + OBJECTS + ["rot"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(layouts)


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":
    random.seed(SEED)

    # Define los rangos de cada archivo: (start_id, end_id)
    batches = [
        (1,   60),
        (61,  120),
        (121, 180),
        (181, 240),
    ]

    for start_id, end_id in batches:
        layouts = generate_layouts(start_id, end_id)
        output_file = Path(OUTPUT_DIR) / f"so101_layouts_{start_id:03d}_{end_id:03d}.csv"
        save_to_csv(layouts, output_file)
        print(f"CSV generado: {output_file.resolve()} ({end_id - start_id + 1} episodios, IDs {start_id}–{end_id})")
