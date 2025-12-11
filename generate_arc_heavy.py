import random
import re
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from visual_intelligence.tasks.json_arc import process_arc_like_iterator

# from utils.unified_constants import DATASET_PATH, DOWNLOADS_PATH

DATASET_NAME = "heavy200k"


def extract_concepts_and_description(code: str) -> dict:
    concepts_match = re.search(r"# concepts:\s*\n((?:#.*\n)+)", code)
    description_match = re.search(r"# description:\s*\n((?:#.*\n)+)", code)

    concepts = []
    if concepts_match:
        concepts_text = re.sub(
            r"^#\s*", "", concepts_match.group(1), flags=re.MULTILINE
        )
        concepts = [c.strip() for c in concepts_text.split(",") if c.strip()]

    description = ""
    if description_match:
        description = re.sub(
            r"^#\s*", "", description_match.group(1), flags=re.MULTILINE
        ).strip()

    return {"concepts": concepts, "description": description}


def convert_heavy200k_to_arc_format_iterator(
    dataset_iterator,
    min_train_examples: int = 2,
    max_train_examples: int = 5,
    num_test_examples: int = 1,
    tasks_per_seed: int = 1,
) -> Iterator[dict[str, Any]]:
    """
    Convert HEAVY200k dataset format to ARC-like format.

    Yields dictionaries with 'train', 'test', and metadata keys.

    Parameters:
        dataset_iterator: Iterator from the HEAVY200k dataset
        min_train_examples: Minimum number of training examples
        max_train_examples: Maximum number of training examples
        num_test_examples: Number of test examples
        tasks_per_seed: Number of different task variations to create per seed example

    Yields:
        Dict with keys: 'train', 'test', 'concepts', 'description', 'source', 'task_id'
    """
    for idx, example_and_metadata in enumerate(dataset_iterator):
        source = example_and_metadata["source"]
        examples = example_and_metadata["examples"]  # List of [input, output] pairs

        # Extract metadata from source code
        metadata = extract_concepts_and_description(source)
        if not metadata["concepts"] or not metadata["description"]:
            print(f"Warning: No metadata found for task {idx}, skipping.")
            continue

        # Check if we have enough examples
        required_examples = min_train_examples + num_test_examples
        if len(examples) < required_examples:
            print(
                f"Warning: Task {idx} has only {len(examples)} examples, need at least {required_examples}, skipping."
            )
            continue

        # Generate multiple tasks from the same seed
        for task_variant in range(tasks_per_seed):
            # Sample examples for train and test
            num_train = random.randint(
                min_train_examples,
                min(max_train_examples, len(examples) - num_test_examples),
            )
            sampled_examples = random.sample(examples, num_train + num_test_examples)

            # Split into train and test
            train_examples = sampled_examples[:num_train]
            test_examples = sampled_examples[num_train:]

            # Convert to ARC format
            train_data = [{"input": inp, "output": out} for inp, out in train_examples]
            test_data = [{"input": inp, "output": out} for inp, out in test_examples]

            # Yield in ARC-like format with all metadata
            yield {
                "train": train_data,
                "test": test_data,
                "concepts": metadata["concepts"],
                "description": metadata["description"],
                "source": source,
                "task_id": f"{idx:06d}_v{task_variant:02d}",
                "seed_id": idx,
                "variant_id": task_variant,
            }


if __name__ == "__main__":

    dataset = load_dataset(
        "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
        # cache_dir=str(DOWNLOADS_PATH / DATASET_NAME),
        streaming=True,
    )

    # There only exists one split (train)
    dataset_iterator = iter(dataset["train"])
    dataset_iterator = convert_heavy200k_to_arc_format_iterator(
        dataset_iterator=dataset_iterator,
        tasks_per_seed=3,
    )

    output_path = Path("datasets/arc-tasks/arc-heavy200k")
    process_arc_like_iterator(
        dataset_iterator=dataset_iterator, output_path=output_path
    )
