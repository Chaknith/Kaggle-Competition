from pathlib import Path

import numpy as np
import pandas as pd

def get_sensor_name(prediction_file: Path) -> str:
    """
    Determine the submission sensor prefix from the prediction file path.

    Checks rednir before nir because 'rednir' also contains 'nir'.
    """
    path_text = prediction_file.as_posix().lower()

    if "rednir" in path_text:
        return "rednir"
    if "nir" in path_text:
        return "nir"
    if "vis" in path_text:
        return "vis"

    raise ValueError(
        f"Could not determine sensor type from path: {prediction_file}"
    )

def create_submission_from_sample(
    predictions_root: Path,
    sample_submission_path: Path,
    output_path: Path,
) -> None:
    """
    Fill the official sample submission using prediction TXT files.

    Expected prediction files:
        results/csrt/HSI-NIR-FalseColor/bee2.txt
        results/csrt/HSI-RedNIR-FalseColor/pills7.txt
        results/csrt/HSI-VIS-FalseColor/dog.txt

    Expected TXT contents:
        x    y    width    height

    Expected submission IDs:
        nir-bee2_1
        rednir-pills7_1
        vis-dog_1
    """

    sample_submission = pd.read_csv(sample_submission_path)

    required_columns = ["ID", "x", "y", "width", "height"]

    if sample_submission.columns.tolist() != required_columns:
        raise ValueError(
            f"Unexpected sample-submission columns: "
            f"{sample_submission.columns.tolist()}\n"
            f"Expected: {required_columns}"
        )

    prediction_files = sorted(predictions_root.rglob("*.txt"))

    if not prediction_files:
        raise FileNotFoundError(
            f"No TXT prediction files found in {predictions_root}"
        )

    # Maps submission IDs to predicted boxes.
    #
    # Example:
    # "nir-bee2_1" -> [x, y, width, height]
    prediction_lookup = {}

    for prediction_file in prediction_files:
        predictions = np.loadtxt(
            prediction_file,
            delimiter="\t",
            dtype=np.float32,
        )

        predictions = np.atleast_2d(predictions)

        if predictions.shape[1] != 4:
            raise ValueError(
                f"{prediction_file} has shape {predictions.shape}. "
                "Expected four columns: x, y, width, height."
            )

        if not np.isfinite(predictions).all():
            raise ValueError(
                f"{prediction_file} contains NaN or infinite values."
            )

        sensor = get_sensor_name(prediction_file)
        sequence_name = prediction_file.stem

        # Example: nir-bee2
        sequence_id = f"{sensor}-{sequence_name}"

        for frame_number, predicted_box in enumerate(
            predictions,
            start=1,
        ):
            submission_id = f"{sequence_id}_{frame_number}"

            if submission_id in prediction_lookup:
                raise ValueError(
                    f"Duplicate prediction ID generated: {submission_id}"
                )

            prediction_lookup[submission_id] = predicted_box

        print(
            f"Loaded {sequence_id}: "
            f"{len(predictions)} frame predictions"
        )

    missing_ids = [
        submission_id
        for submission_id in sample_submission["ID"]
        if submission_id not in prediction_lookup
    ]

    if missing_ids:
        print(f"\nMissing predictions: {len(missing_ids)}")
        print("First missing IDs:")

        for submission_id in missing_ids[:20]:
            print(f"  {submission_id}")

        raise ValueError(
            "Some sample-submission IDs do not have corresponding "
            "predictions."
        )

    sample_ids = set(sample_submission["ID"])
    extra_ids = set(prediction_lookup) - sample_ids

    if extra_ids:
        print(
            f"Warning: {len(extra_ids)} generated prediction IDs "
            "are not present in the sample submission."
        )

        for submission_id in sorted(extra_ids)[:20]:
            print(f"  Extra: {submission_id}")

    # Fill each row while preserving the official submission order.
    for row_index, submission_id in enumerate(
        sample_submission["ID"]
    ):
        x, y, width, height = prediction_lookup[submission_id]

        sample_submission.loc[
            row_index,
            ["x", "y", "width", "height"],
        ] = [x, y, width, height]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_submission.to_csv(
        output_path,
        index=False,
        float_format="%.3f",
    )

    print()
    print(f"Prediction files: {len(prediction_files)}")
    print(f"Submission rows: {len(sample_submission)}")
    print(f"Submission saved to: {output_path}")
    print()
    print(sample_submission.head())

predictions_root = Path("results/csrt")

sample_submission_path = Path("../data/validation/sample_submission.csv")
output_path = Path("results/submission.csv")

create_submission_from_sample(
    predictions_root=predictions_root,
    sample_submission_path=sample_submission_path,
    output_path=output_path,
)