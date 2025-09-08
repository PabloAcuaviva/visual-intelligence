from visual_logic.dataset_generation.registry import DATASET_GENERATORS

if __name__ == "__main__":
    subset_sizes = [3, 5, 10, 30, 50, 100, 300, 500, 1000]

    n_test = 200

    # Call the desired dataset generators by name
    # Example: DATASET_GENERATORS['maze']()
    #          DATASET_GENERATORS['navigation2d']()
    #          DATASET_GENERATORS['tower_of_hanoi']()
    #          DATASET_GENERATORS['gol']()
    #          DATASET_GENERATORS['langton_ant']()
    # Uncomment or add as needed:
    # DATASET_GENERATORS['maze']()

    # DATASET_GENERATORS['navigation2d']()

    # DATASET_GENERATORS["navigation2d_any_to_any"](subset_sizes=subset_sizes)
    # DATASET_GENERATORS["maze_small"](subset_sizes=subset_sizes)
    # DATASET_GENERATORS["maze"](subset_sizes=subset_sizes)

    # DATASET_GENERATORS["shortest_path"](
    #     subset_sizes=[3, 5, 10, 30, 50, 100, 250, 500, 1000]
    # )
    # for steps in [1, 2]:  # 3, 5, 10]:
    #     DATASET_GENERATORS["gol"](steps=steps, subset_sizes=subset_sizes)

    # for steps in [2, 3, 5, 10, 20, 30, 50]:
    #     DATASET_GENERATORS["langton_ant"](
    #         steps=steps,
    #         subset_sizes=subset_sizes,
    #         n_train=max(subset_sizes),
    #         n_test=n_test,
    #     )

    # DATASET_GENERATORS["connect4"](subset_sizes=subset_sizes)
    # DATASET_GENERATORS["general_hanoi"](steps=1, subset_sizes=subset_sizes)

    # DATASET_GENERATORS["tower_of_hanoi"]()
    # DATASET_GENERATORS["chess_meta_in_n"](
    #     subset_sizes=subset_sizes,
    #     n_train=max(subset_sizes),
    #     n_test=n_test,
    # )

    # for rule in [30, 90, 111]:
    #     DATASET_GENERATORS["cellular_automata_1d"](
    #         rule=rule,
    #         subset_sizes=subset_sizes,
    #         n_train=max(subset_sizes),
    #         n_test=n_test,
    #     )

    # DATASET_GENERATORS["sudoku"](subset_sizes=subset_sizes)
    # DATASET_GENERATORS["sudoku"](variant="mini", subset_sizes=subset_sizes)

    # DATASET_GENERATORS["hitori"](subset_sizes=subset_sizes)

    # Some interesting Cellular Automata to explore and their Wolfram Class

    # eca_classes = {
    #     "Class 2": [4, 108, 170, 250],
    #     "Class 3": [30, 45, 90, 150],
    #     "Class 4": [110, 54, 62, 106],
    #     "Class 1": [8, 32, 128, 160],
    # }
    # subset_sizes = [3, 5, 10, 30, 50, 100, 300, 500, 1000, 2000, 5000]
    # for class_rules in eca_classes.values():
    #     for rule in class_rules:
    #         DATASET_GENERATORS["cellular_automata_1d"](
    #             rule=rule,
    #             subset_sizes=subset_sizes,
    #             n_train=max(subset_sizes),
    #             n_test=n_test,
    #         )

    ###
    # EXTENDED DATASETS
    ###
    vdm_datasets_path = "datasets/data"
    subset_sizes = [3, 5, 10, 30, 50, 100, 300, 500, 1000, 3000, 5000]
    gol_rules = {
        # "Gol_DayAndNight": {"birth_rule": [3, 6, 7, 8], "survival_rule": [3, 4, 6, 7, 8]},
        # "Gol_Maze": {"birth_rule": [3], "survival_rule": [1, 2, 3, 4, 5]},
        "Gol_Seeds": {"birth_rule": [2], "survival_rule": []},
        "Gol_Life_B3S2": {"birth_rule": [3], "survival_rule": [2]},
    }
    for gol_variant_name, rules in gol_rules.items():
        DATASET_GENERATORS["gol"](
            steps=1,
            gol_variant_name=gol_variant_name,
            n_train=max(subset_sizes),
            subset_sizes=subset_sizes,
            n_test=n_test,
            distance_threshold=0.25 if gol_variant_name != "Seeds" else 0.15,
            **rules,
        )
    # DATASET_GENERATORS["sudoku"](
    #     n_train=max(subset_sizes),
    #     subset_sizes=subset_sizes,
    #     n_test=n_test,
    #     extend_dataset=f"{vdm_datasets_path}/sudoku_standard_easy",
    # )
    # DATASET_GENERATORS["sudoku"](
    #     variant="mini",
    #     n_train=max(subset_sizes),
    #     subset_sizes=subset_sizes,
    #     n_test=n_test,
    #     extend_dataset=f"{vdm_datasets_path}/sudoku_mini_easy",
    # )
    # DATASET_GENERATORS["navigation2d_any_to_any"](
    #     n_train=max(subset_sizes),
    #     subset_sizes=subset_sizes,
    #     n_test=n_test,
    #     extend_dataset=f"{vdm_datasets_path}/navigation2d_any_to_any",
    # )
    # DATASET_GENERATORS["maze_small"](
    #     n_train=100_000,
    #     subset_sizes=subset_sizes + [10_000, 20_000, 50_000, 100_000],
    #     n_test=0,
    #     extend_dataset=f"{vdm_datasets_path}/maze_small",
    # )
    # DATASET_GENERATORS["maze"](
    #     n_train=max(subset_sizes),
    #     subset_sizes=subset_sizes,
    #     n_test=n_test,
    #     extend_dataset=f"{vdm_datasets_path}/maze",
    # )
    # DATASET_GENERATORS["hitori"](
    #     n_train=max(subset_sizes),
    #     subset_sizes=subset_sizes,
    #     n_test=n_test,
    #     extend_dataset=f"{vdm_datasets_path}/hitori_5_easy",
    # )

    # for steps in [10]:  # [2, 3, 5]:
    #     DATASET_GENERATORS["langton_ant"](
    #         steps=steps,
    #         subset_sizes=subset_sizes,
    #         n_train=max(subset_sizes),
    #         n_test=n_test,
    #         extend_dataset=f"{vdm_datasets_path}/langton_ant_step{steps}",
    #     )

    # DATASET_GENERATORS["connect4"](
    #     n_train=max(subset_sizes),
    #     subset_sizes=subset_sizes,
    #     n_test=100,
    #     extend_dataset=f"{vdm_datasets_path}/connect4",
    # )
