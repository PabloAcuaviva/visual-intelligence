import random
from pathlib import Path

from visual_intelligence.dataset_generation import VideoConfig
from visual_intelligence.dataset_generation.registry import DATASET_GENERATORS
from visual_intelligence.tasks.render.schemas import RenderStyle

if __name__ == "__main__":
    datasets_path = Path("datasets/intermediate_states")
    random.seed(42)  # For reproducibility
    n_test = 50
    subset_sizes = [1000]

    IMAGE_HEIGHT = IMAGE_WIDTH = 256

    # DATASET_GENERATORS["hitori"](
    #     n_train=max(subset_sizes),
    #     subset_sizes=subset_sizes,
    #     n_test=n_test,
    #     out_dir=datasets_path,
    # )

    # DATASET_GENERATORS["sudoku"](
    #     n_train=max(subset_sizes),
    #     subset_sizes=subset_sizes,
    #     n_test=n_test,
    #     out_dir=datasets_path,
    # )

    # DATASET_GENERATORS["sudoku"](
    #     variant="mini",
    #     n_train=max(subset_sizes),
    #     subset_sizes=subset_sizes,
    #     n_test=n_test,
    #     out_dir=datasets_path,
    # )

    eca_classes = {
        "Class 2": [4, 108, 170, 250],
        # "Class 3": [30, 45, 90, 150],
        # "Class 4": [110, 54, 62, 106],
        # "Class 1": [8, 32, 128, 160],
    }
    # DATASET_GENERATORS["cellular_automata_1d"](
    #     rule=rule,
    #     subset_sizes=subset_sizes,
    #     n_train=max(subset_sizes),
    #     n_test=n_test,
    #     out_dir=datasets_path,
    # )

    ###
    # Studied GOL rules
    ###

    gol_rule = {
        # --- The "Life" Family ---
        "Life": "B3/S23",
        "HighLife": "B36/S23",
        "EightLife": "B3/S238",
        "PedestrianLife": "B38/S23",
        "HoneyLife": "B38/S238",
        "DryLife": "B37/S23",
        "PseudoLife": "B357/S238",
        "LongLife": "B345/S5",
        "LowDeath": "B368/S238",
        "3-4 Life": "B34/S34",
        # --- Spaceships & Movement ---
        "Morley": "B368/S245",
        "2x2": "B36/S125",
        "DayAndNight": "B3678/S34678",
        "StarWars": "B2/S345",
        "GliderLife": "B36/S235",
        "Flock": "B3/S12",
        # --- Structure & Textures ---
        "Maze": "B3/S12345",
        "Mazectric": "B3/S1234",
        "Coral": "B3/S45678",
        "WalledCities": "B45678/S2345",
        "Coagulations": "B378/S235678",
        "LandRush": "B35/S234578",
        "Holstein": "B35678/S4678",
        "Bacteria": "B34/S456",
        "Iceballs": "B25678/S5678",
        # --- Biology & Liquids ---
        "Amoeba": "B357/S1358",
        "Diamoeba": "B35678/S5678",
        "Assimilation": "B345/S4567",
        "Stains": "B3678/S235678",
        "SlowBlob": "B367/S125678",
        "Corrosion": "B3/S124",
        "Bugs": "B3567/S15678",
        # --- Explosive & Mathematical ---
        "Replicator": "B1357/S1357",
        "Seeds": "B2/S",
        "Serviettes": "B234/S",
        "Gnarl": "B1/S1",
        "LiveFreeOrDie": "B2/S0",
        "JustFriends": "B2/S34",
        "Electra": "B1/S12",
        "Spirals": "B2/S",
        # --- Utility ---
        "Vote": "B5678/S45678",
        "Anneal": "B4678/S35678",
        "Gems": "B3457/S4568",
    }

    gol_rules = {
        name: {
            "birth_rule": list(map(int, rules.split("/")[0][1:])),
            "survival_rule": list(map(int, rules.split("/")[1][1:])),
        }
        for name, rules in gol_rule.items()
    }

    gol_n_grid = 12
    gol_style = RenderStyle(
        cell_size=20,
        grid_border_size=1,
        value_to_color={
            0: (0, 0, 0),  # Black
            1: (0, 116, 217),  # Blue
            2: (255, 65, 54),  # Red
            3: (46, 204, 64),  # Green
            4: (255, 220, 0),  # Yellow
            5: (170, 170, 170),  # Grey
            6: (240, 18, 190),  # Fuchsia
            7: (255, 133, 27),  # Orange
            8: (127, 219, 255),  # Teal
            9: (135, 12, 37),  # Brown
            10: (163, 73, 164),  # Purple (deep violet)
            11: (255, 182, 193),  # Pink (light pink)
            12: (0, 255, 255),  # Cyan (bright aqua)
            13: (128, 0, 128),  # Dark purple
            14: (192, 192, 192),  # Silver (light gray)
        },
        background_color=(255, 255, 255),  # White background
        border_color=(85, 85, 85),  # Medium gray border
    )
    for gol_variant_name, rules in gol_rules.items():
        DATASET_GENERATORS["gol"](
            steps=10,
            gol_variant_name=gol_variant_name,
            n_train=max(subset_sizes),
            subset_sizes=subset_sizes,
            n_test=n_test,
            width=gol_n_grid,
            height=gol_n_grid,
            distance_threshold=0.15,
            style=gol_style,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            out_dir=datasets_path / "gol_generalizations" / "known_variants",
            video=VideoConfig(
                fps=4,
                frames_per_init=5,
                frames_per_intermediate=4,
                frames_per_target=4,
            ),
            **rules,
        )

    ###
    # Random GOL rules
    ###

    # Generate a bunch of these random ones (B*/S*) make sure they are not in known
    # Distribution: how many variants should have N digits in birth/survival rules
    # Keys are digit counts (1-8), values are target number of variants with that count
    VARIANTS_CONFIG = {
        "birth": {1: 15, 2: 35, 3: 25, 4: 10, 5: 5},  # More weight on 1-3 digits
        "survival": {
            0: 5,
            1: 10,
            2: 30,
            3: 30,
            4: 15,
            5: 10,
        },  # More weight on 2-4 digits
    }

    # Extract known rules as frozensets for comparison
    known_rules = {
        (frozenset(rules["birth_rule"]), frozenset(rules["survival_rule"]))
        for rules in gol_rules.values()
    }

    # Build weighted lists for sampling counts independently
    birth_counts = [
        count
        for count, weight in VARIANTS_CONFIG["birth"].items()
        for _ in range(weight)
    ]
    survival_counts = [
        count
        for count, weight in VARIANTS_CONFIG["survival"].items()
        for _ in range(weight)
    ]

    random_gol_rules = {}
    attempts = 0
    max_attempts = 10000
    total_variants = 200
    while len(random_gol_rules) < total_variants and attempts < max_attempts:
        attempts += 1
        # Sample counts independently from each distribution
        birth_count = random.choice(birth_counts)
        survival_count = random.choice(survival_counts)

        # Randomly select which digits to include (from 0-8)
        birth_rule = sorted(random.sample(range(9), birth_count))
        survival_rule = (
            sorted(random.sample(range(9), survival_count))
            if survival_count > 0
            else []
        )

        # Skip if birth is empty (nothing can be born)
        if not birth_rule:
            continue

        # Check if this rule is already known or already generated
        rule_key = (frozenset(birth_rule), frozenset(survival_rule))
        if rule_key in known_rules:
            continue
        if rule_key in {
            (frozenset(r["birth_rule"]), frozenset(r["survival_rule"]))
            for r in random_gol_rules.values()
        }:
            continue

        # Create name in B*/S* format
        birth_str = "".join(map(str, birth_rule))
        survival_str = "".join(map(str, survival_rule))
        variant_name = f"B{birth_str}/S{survival_str}"

        random_gol_rules[variant_name] = {
            "birth_rule": birth_rule,
            "survival_rule": survival_rule,
        }

    # Generate
    for gol_variant_name, rules in random_gol_rules.items():
        DATASET_GENERATORS["gol"](
            steps=10,
            gol_variant_name=gol_variant_name,
            n_train=max(subset_sizes),
            subset_sizes=subset_sizes,
            n_test=n_test,
            distance_threshold=0.15,
            style=gol_style,
            width=gol_n_grid,
            height=gol_n_grid,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            out_dir=datasets_path / "gol_generalizations" / "random_variants",
            video=VideoConfig(
                fps=4,
                frames_per_init=5,
                frames_per_intermediate=4,
                frames_per_target=4,
            ),
            **rules,
        )

    # # Langton ant
    # for steps in [2, 3, 5, 10]:
    #     DATASET_GENERATORS["langton_ant"](
    #         steps=steps,
    #         subset_sizes=subset_sizes,
    #         n_train=max(subset_sizes),
    #         n_test=n_test,
    #         out_dir=datasets_path,
    #     )

    ###
    # Navigation and mazes
    ###
    # DATASET_GENERATORS["navigation2d_any_to_any"](
    #     n_train=max(subset_sizes),
    #     subset_sizes=subset_sizes,
    #     n_test=n_test,
    #     out_dir=datasets_path,
    # )
    # DATASET_GENERATORS["maze_small"](
    #     n_train=max(subset_sizes),
    #     subset_sizes=subset_sizes,
    #     n_test=0,
    #     out_dir=datasets_path,
    # )
    # DATASET_GENERATORS["maze"](
    #     n_train=max(subset_sizes),
    #     subset_sizes=subset_sizes,
    #     n_test=n_test,
    #     out_dir=datasets_path,
    # )
    # )
