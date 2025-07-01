from visual_logic.dataset_generation.registry import DATASET_GENERATORS

if __name__ == "__main__":
    # Call the desired dataset generators by name
    # Example: DATASET_GENERATORS['maze']()
    #          DATASET_GENERATORS['navigation2d']()
    #          DATASET_GENERATORS['tower_of_hanoi']()
    #          DATASET_GENERATORS['gol']()
    #          DATASET_GENERATORS['langton_ant']()
    # Uncomment or add as needed:
    # DATASET_GENERATORS['maze']()
    # DATASET_GENERATORS['navigation2d']()
    # DATASET_GENERATORS['gol']()
    # DATASET_GENERATORS['langton_ant']()
    DATASET_GENERATORS["tower_of_hanoi"]()
