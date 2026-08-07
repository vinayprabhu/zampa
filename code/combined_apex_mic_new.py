"""Configurable MIC inference for the released APEX 1.0 and APEX 1.1 models.

The public APEX 1.0 repository predicts MICs for 34 strains, not 40. The
public APEX 1.1 repository predicts 11 strains. The default ``mode="both"``
preserves the original 45-prediction output per peptide:

* one peptide -> a flat ``list[float]`` of length 45
* multiple peptides -> an ``n x 45`` pandas DataFrame

Other modes expose either full model, the matched 11-strain APEX 1.0 subset,
or one matched-panel median per model while preserving the same list/DataFrame
return convention.

MIC values are in micromolar (uM), matching the upstream repositories.

Example
-------
from combined_apex_mic import predict_apex_mics

result = predict_apex_mics(
    ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"],
    apex10_repo="/path/to/apex",
    apex11_repo="/path/to/apex-pathogen",
    mode="both",
)
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Literal, Sequence

import numpy as np


APEX_10_STRAINS = [
    "E. coli ATCC11775",
    "P. aeruginosa PAO1",
    "P. aeruginosa PA14",
    "S. aureus ATCC12600",
    "E. coli AIG221",
    "E. coli AIG222",
    "K. pneumoniae ATCC13883",
    "A. baumannii ATCC19606",
    "A. muciniphila ATCC BAA-835",
    "B. fragilis ATCC25285",
    "B. vulgatus ATCC8482",
    "C. aerofaciens ATCC25986",
    "C. scindens ATCC35704",
    "B. thetaiotaomicron ATCC29148",
    "B. thetaiotaomicron Complemmented",
    "B. thetaiotaomicron Mutant",
    "B. uniformis ATCC8492",
    "B. eggerthi ATCC27754",
    "C. spiroforme ATCC29900",
    "P. distasonis ATCC8503",
    "P. copri DSMZ18205",
    "B. ovatus ATCC8483",
    "E. rectale ATCC33656",
    "C. symbiosum",
    "R. obeum",
    "R. torques",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC700802",
    "vancomycin-resistant E. faecium ATCC700221",
    "E. coli Nissle",
    "Salmonella enterica ATCC 9150 (BEIRES NR-515)",
    "Salmonella enterica (BEIRES NR-170)",
    "Salmonella enterica ATCC 9150 (BEIRES NR-174)",
    "L. monocytogenes ATCC 19111 (BEIRES NR-106)",
]

APEX_11_STRAINS = [
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
]

APEX_10_COLUMNS = [f"APEX_1.0 | {strain}" for strain in APEX_10_STRAINS]
APEX_11_COLUMNS = [f"APEX_1.1 | {strain}" for strain in APEX_11_STRAINS]
OUTPUT_COLUMNS = APEX_10_COLUMNS + APEX_11_COLUMNS

# Python indices of the APEX 1.0 outputs corresponding to the 11 strains
# represented by APEX 1.1. The order below is intentionally the user-supplied
# APEX 1.0 index order; order does not affect the matched-panel median.
APEX_10_MATCHED_11_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 26, 27, 28)
APEX_10_MATCHED_11_COLUMNS = [
    APEX_10_COLUMNS[index] for index in APEX_10_MATCHED_11_INDICES
]
MEDIAN_COLUMNS = [
    "APEX_1.0 | median MIC across matched 11 strains",
    "APEX_1.1 | median MIC across 11 strains",
]
VALID_MODES = frozenset({"all_1", "all_pathogen", "both", "subset_1", "median"})

_VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_AA_TO_INDEX = {
    "A": 3,
    "C": 4,
    "D": 5,
    "E": 6,
    "F": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "K": 11,
    "L": 12,
    "M": 13,
    "N": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "V": 20,
    "W": 21,
    "Y": 22,
}


def predict_apex_mics(
    sequences: Sequence[str],
    apex10_repo: str | os.PathLike[str],
    apex11_repo: str | os.PathLike[str],
    *,
    mode: Literal["all_1", "all_pathogen", "both", "subset_1", "median"] = "both",
    device: Literal["auto", "cpu", "cuda"] = "auto",
    batch_size: int = 3000,
) -> list[float] | "pandas.DataFrame":
    """Predict MICs with one or both released APEX ensembles.

    Parameters
    ----------
    sequences:
        Non-empty sequence of peptide strings containing only the 20 standard
        amino-acid letters. Peptides may be at most 50 residues long.
    apex10_repo:
        Local checkout of
        https://gitlab.com/machine-biology-group-public/apex
    apex11_repo:
        Local checkout of
        https://gitlab.com/machine-biology-group-public/apex-pathogen
    mode:
        Controls which predictions are returned:

        * ``"all_1"``: all 34 APEX 1.0 strain predictions.
        * ``"all_pathogen"``: all 11 APEX 1.1 strain predictions.
        * ``"both"``: all 45 predictions, with APEX 1.0 first (default).
        * ``"subset_1"``: the 11 APEX 1.0 outputs at indices
          ``[0, 1, 2, 3, 4, 5, 6, 7, 26, 27, 28]``.
        * ``"median"``: the median of that APEX 1.0 subset followed by
          the median of all 11 APEX 1.1 outputs.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU.
    batch_size:
        Number of peptide sequences evaluated per model forward pass.

    Returns
    -------
    list[float] or pandas.DataFrame
        One sequence returns a flat list. Two or more sequences return a
        DataFrame indexed by peptide sequence. The output width is 34, 11, 45,
        11, or 2 for ``all_1``, ``all_pathogen``, ``both``, ``subset_1``, or
        ``median``, respectively. Values are MICs in uM.

    Notes
    -----
    The two upstream repositories both contain a top-level ``utils.py`` and
    save full PyTorch model objects. Each ensemble therefore runs in an isolated
    subprocess to prevent module-name collisions during checkpoint loading.
    """
    normalized = _validate_sequences(sequences)

    if not isinstance(mode, str) or mode not in VALID_MODES:
        choices = ", ".join(repr(choice) for choice in sorted(VALID_MODES))
        raise ValueError(f"mode must be one of: {choices}")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: 'auto', 'cpu', 'cuda'")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")

    pred10 = None
    if mode in {"all_1", "both", "subset_1", "median"}:
        repo10 = _validate_repo(apex10_repo, "APEX 1.0")
        pred10 = _run_isolated_worker(
            model_id="apex10",
            sequences=normalized,
            repo=repo10,
            device=device,
            batch_size=batch_size,
        )
        _check_prediction_shape(
            pred10, len(normalized), len(APEX_10_STRAINS), "APEX 1.0"
        )

    pred11 = None
    if mode in {"all_pathogen", "both", "median"}:
        repo11 = _validate_repo(apex11_repo, "APEX 1.1")
        pred11 = _run_isolated_worker(
            model_id="apex11",
            sequences=normalized,
            repo=repo11,
            device=device,
            batch_size=batch_size,
        )
        _check_prediction_shape(
            pred11, len(normalized), len(APEX_11_STRAINS), "APEX 1.1"
        )

    output, columns = _select_mode_output(mode, pred10, pred11)

    if len(normalized) == 1:
        return output[0].astype(float).tolist()

    import pandas as pd

    frame = pd.DataFrame(output, index=normalized, columns=columns)
    frame.index.name = "peptide_sequence"
    return frame


def _select_mode_output(
    mode: str,
    pred10: np.ndarray | None,
    pred11: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    """Select or aggregate validated model outputs for the requested mode."""
    if mode == "all_1":
        assert pred10 is not None
        return pred10, APEX_10_COLUMNS
    if mode == "all_pathogen":
        assert pred11 is not None
        return pred11, APEX_11_COLUMNS
    if mode == "both":
        assert pred10 is not None and pred11 is not None
        return np.concatenate((pred10, pred11), axis=1), OUTPUT_COLUMNS
    if mode == "subset_1":
        assert pred10 is not None
        return pred10[:, APEX_10_MATCHED_11_INDICES], APEX_10_MATCHED_11_COLUMNS

    assert mode == "median" and pred10 is not None and pred11 is not None
    matched_pred10 = pred10[:, APEX_10_MATCHED_11_INDICES]
    medians = np.column_stack(
        (np.median(matched_pred10, axis=1), np.median(pred11, axis=1))
    )
    return medians, MEDIAN_COLUMNS


def _validate_sequences(sequences: Sequence[str]) -> list[str]:
    if isinstance(sequences, (str, bytes)) or not isinstance(sequences, Sequence):
        raise TypeError("sequences must be a non-empty sequence of peptide strings")
    if len(sequences) == 0:
        raise ValueError("sequences must contain at least one peptide")

    normalized: list[str] = []
    for position, sequence in enumerate(sequences):
        if not isinstance(sequence, str):
            raise TypeError(f"sequence at position {position} is not a string")
        peptide = sequence.strip().upper()
        if not peptide:
            raise ValueError(f"sequence at position {position} is empty")
        if len(peptide) > 50:
            raise ValueError(
                f"sequence at position {position} has {len(peptide)} residues; "
                "the released APEX models accept at most 50"
            )
        invalid = sorted(set(peptide) - _VALID_AMINO_ACIDS)
        if invalid:
            raise ValueError(
                f"sequence at position {position} contains unsupported residues: "
                f"{', '.join(invalid)}"
            )
        normalized.append(peptide)
    return normalized


def _validate_repo(repo: str | os.PathLike[str], label: str) -> Path:
    path = Path(repo).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} repository directory not found: {path}")
    return path


def _run_isolated_worker(
    *,
    model_id: Literal["apex10", "apex11"],
    sequences: list[str],
    repo: Path,
    device: str,
    batch_size: int,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix=f"{model_id}_") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "sequences.json"
        output_path = temp_path / "predictions.npy"
        input_path.write_text(json.dumps(sequences), encoding="utf-8")

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_worker",
            model_id,
            "--repo",
            str(repo),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"{model_id} inference failed:\n{detail}")
        if not output_path.is_file():
            raise RuntimeError(f"{model_id} inference produced no output")
        return np.load(output_path, allow_pickle=False)


def _encode_sequences(sequences: list[str], max_len: int = 52) -> np.ndarray:
    encoded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        tokens = "1" + sequence[: max_len - 2] + "2"
        encoded[row, 0] = 1
        encoded[row, len(tokens) - 1] = 2
        for column, amino_acid in enumerate(tokens[1:-1], start=1):
            encoded[row, column] = _AA_TO_INDEX[amino_acid]
    return encoded


def _torch_load(torch_module, checkpoint: Path, device):
    kwargs = {"map_location": device}
    try:
        return torch_module.load(checkpoint, weights_only=False, **kwargs)
    except TypeError:
        # PyTorch 1.11, used by the original repositories, predates
        # the weights_only keyword.
        return torch_module.load(checkpoint, **kwargs)


def _resolve_device(torch_module, requested: str):
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch_module.device(requested)


def _predict_ensemble(
    *,
    models,
    encoded: np.ndarray,
    torch_module,
    device,
    batch_size: int,
) -> np.ndarray:
    if not models:
        raise RuntimeError("No pretrained models were found")

    prediction_sum = None
    with torch_module.inference_mode():
        for model in models:
            model = model.to(device).eval()
            batches = []
            for start in range(0, len(encoded), batch_size):
                tensor = torch_module.as_tensor(
                    encoded[start : start + batch_size],
                    dtype=torch_module.long,
                    device=device,
                )
                transformed = model(tensor).detach().cpu().numpy()
                batches.append(np.power(10.0, 6.0 - transformed))
            prediction = np.concatenate(batches, axis=0)
            prediction_sum = (
                prediction.astype(np.float64, copy=True)
                if prediction_sum is None
                else prediction_sum + prediction
            )
            model.to("cpu")
            del model
            if device.type == "cuda":
                torch_module.cuda.empty_cache()
    return prediction_sum / float(len(models))


def _apex10_worker(repo: Path, sequences: list[str], requested_device: str, batch_size: int):
    model_module = repo / "AMP_DL_model_twohead.py"
    key_file = repo / "best_key_list"
    model_dir = repo / "trained_models"
    for required in (model_module, key_file, model_dir):
        if not required.exists():
            raise FileNotFoundError(f"Required APEX 1.0 path not found: {required}")

    sys.path.insert(0, str(repo))
    importlib.import_module("AMP_DL_model_twohead")
    import torch

    device = _resolve_device(torch, requested_device)
    model_keys = [
        line.strip()
        for line in key_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoints = [
        model_dir / f"trained_all_model_{model_key}_ensemble_{repeat}"
        for model_key in model_keys
        for repeat in range(5)
    ]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing APEX 1.0 checkpoints, beginning with: " + ", ".join(missing[:3])
        )
    models = [_torch_load(torch, path, device) for path in checkpoints]
    return _predict_ensemble(
        models=models,
        encoded=_encode_sequences(sequences),
        torch_module=torch,
        device=device,
        batch_size=batch_size,
    )


def _apex11_worker(repo: Path, sequences: list[str], requested_device: str, batch_size: int):
    model_module = repo / "APEX_models.py"
    model_dir = repo / "APEX_pathogen_models"
    for required in (model_module, model_dir):
        if not required.exists():
            raise FileNotFoundError(f"Required APEX 1.1 path not found: {required}")

    sys.path.insert(0, str(repo))
    importlib.import_module("APEX_models")
    import torch

    device = _resolve_device(torch, requested_device)
    checkpoints = sorted(
        path for path in model_dir.glob("APEX_*") if path.is_file()
    )
    models = [_torch_load(torch, path, device) for path in checkpoints]
    return _predict_ensemble(
        models=models,
        encoded=_encode_sequences(sequences),
        torch_module=torch,
        device=device,
        batch_size=batch_size,
    )


def _check_prediction_shape(
    predictions: np.ndarray,
    expected_rows: int,
    expected_columns: int,
    label: str,
) -> None:
    expected = (expected_rows, expected_columns)
    if predictions.shape != expected:
        raise RuntimeError(
            f"{label} returned shape {predictions.shape}; expected {expected}. "
            "The checkpoint and repository code may be mismatched."
        )


def _worker_main(arguments: argparse.Namespace) -> None:
    sequences = json.loads(arguments.input.read_text(encoding="utf-8"))
    if arguments.worker == "apex10":
        predictions = _apex10_worker(
            arguments.repo, sequences, arguments.device, arguments.batch_size
        )
    else:
        predictions = _apex11_worker(
            arguments.repo, sequences, arguments.device, arguments.batch_size
        )
    np.save(arguments.output, predictions, allow_pickle=False)


def _parse_internal_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_worker", dest="worker", choices=("apex10", "apex11"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=3000)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_internal_arguments()
    if args.worker is None:
        raise SystemExit("Import this module and call predict_apex_mics().")
    _worker_main(args)
