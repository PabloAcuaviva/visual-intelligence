from visual_intelligence.dataset_generation.registry import DATASET_GENERATORS

if __name__ == "__main__":
    datasets_path = "datasets/extended-data"
    n_test = 200
    subset_sizes = [3, 5, 10, 30, 50, 100, 300, 500, 1000, 3000, 5000]

    generate_games = True
    generate_automatas = True
    generate_navigation = True

    ###
    # Games
    ###
    if generate_games:
        DATASET_GENERATORS["hitori"](
            n_train=max(subset_sizes),
            subset_sizes=subset_sizes,
            n_test=n_test,
            out_dir=datasets_path,
        )

        # Sudoku variants
        DATASET_GENERATORS["sudoku"](
            n_train=max(subset_sizes),
            subset_sizes=subset_sizes,
            n_test=n_test,
            out_dir=datasets_path,
        )

        DATASET_GENERATORS["sudoku"](
            variant="mini",
            n_train=max(subset_sizes),
            subset_sizes=subset_sizes,
            n_test=n_test,
            out_dir=datasets_path,
        )

        DATASET_GENERATORS["connect4"](
            n_train=max(subset_sizes),
            subset_sizes=subset_sizes,
            n_test=n_test,
            out_dir=datasets_path,
        )

        DATASET_GENERATORS["chess_mate_in_n"](
            subset_sizes=[s for s in subset_sizes if s <= 2000],  # Dataset bottleneck
            n_train=max(subset_sizes),
            n_test=n_test,
            out_dir=datasets_path,
        )

    ###
    # Automatas
    ###
    # Cellular Automata with Wolfram Class
    if generate_automatas:
        eca_classes = {
            "Class 2": [4, 108, 170, 250],
            "Class 3": [30, 45, 90, 150],
            "Class 4": [110, 54, 62, 106],
            "Class 1": [8, 32, 128, 160],
        }
        for class_rules in eca_classes.values():
            for rule in class_rules:
                DATASET_GENERATORS["cellular_automata_1d"](
                    rule=rule,
                    subset_sizes=subset_sizes,
                    n_train=max(subset_sizes),
                    n_test=n_test,
                    out_dir=datasets_path,
                )

        # Game of Life variants
        gol_rules = {
            "Gol_DayAndNight": {
                "birth_rule": [3, 6, 7, 8],
                "survival_rule": [3, 4, 6, 7, 8],
            },
            "Gol_Maze": {"birth_rule": [3], "survival_rule": [1, 2, 3, 4, 5]},
            "Gol_Seeds": {"birth_rule": [2], "survival_rule": []},
            "Gol_Life": {"birth_rule": [3], "survival_rule": [2, 3]},
            "Gol_B3_S2": {"birth_rule": [3], "survival_rule": [2]},
        }
        for gol_variant_name, rules in gol_rules.items():
            DATASET_GENERATORS["gol"](
                steps=1,
                gol_variant_name=gol_variant_name,
                n_train=max(subset_sizes),
                subset_sizes=subset_sizes,
                n_test=n_test,
                distance_threshold=0.25 if gol_variant_name != "Seeds" else 0.15,
                out_dir=datasets_path,
                **rules,
            )

        # Langton ant
        for steps in [2, 3, 5, 10]:
            DATASET_GENERATORS["langton_ant"](
                steps=steps,
                subset_sizes=subset_sizes,
                n_train=max(subset_sizes),
                n_test=n_test,
                out_dir=datasets_path,
            )

    ###
    # Navigation and mazes
    ###
    if generate_navigation:
        DATASET_GENERATORS["navigation2d_any_to_any"](
            n_train=max(subset_sizes),
            subset_sizes=subset_sizes,
            n_test=n_test,
            out_dir=datasets_path,
        )
        DATASET_GENERATORS["maze_small"](
            n_train=max(subset_sizes),
            subset_sizes=subset_sizes,
            n_test=0,
            out_dir=datasets_path,
        )
        DATASET_GENERATORS["maze"](
            n_train=max(subset_sizes),
            subset_sizes=subset_sizes,
            n_test=n_test,
            out_dir=datasets_path,
        )
