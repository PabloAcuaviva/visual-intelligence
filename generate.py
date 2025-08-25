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
    # DATASET_GENERATORS["maze_small"](subset_sizes=[10, 1000])
    # DATASET_GENERATORS['navigation2d']()

    # DATASET_GENERATORS["navigation2d_any_to_any"](
    #     subset_sizes=[3, 5, 10, 30, 50, 100, 200, 1000]
    # )
    # DATASET_GENERATORS["shortest_path"](
    #     subset_sizes=[3, 5, 10, 30, 50, 100, 250, 500, 1000]
    # )
    # for steps in [1, 2]:  # 3, 5, 10]:
    #     DATASET_GENERATORS["gol"](steps=steps, subset_sizes=subset_sizes)

    for steps in [2, 3, 5, 10, 20, 30, 50]:
        DATASET_GENERATORS["langton_ant"](
            steps=steps,
            subset_sizes=subset_sizes,
            n_train=max(subset_sizes),
            n_test=n_test,
        )

    # DATASET_GENERATORS["connect4"](subset_sizes=subset_sizes)
    # DATASET_GENERATORS["general_hanoi"](steps=1, subset_sizes=subset_sizes)

    # DATASET_GENERATORS["tower_of_hanoi"]()

    # DATASET_GENERATORS["sudoku"](subset_sizes=subset_sizes)
    # DATASET_GENERATORS["sudoku"](variant="mini", subset_sizes=subset_sizes)

    # DATASET_GENERATORS["hitori"](subset_sizes=subset_sizes)
    # DATASET_GENERATORS["nurikabe"]()
