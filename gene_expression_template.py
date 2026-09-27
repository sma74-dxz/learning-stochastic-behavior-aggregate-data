"""Template for the paper's single-cell RNA-seq aggregate experiment.

Expected authorized local files:
    D0.npy
    D2.npy
    D4.npy
    D7.npy

Each array should have shape [number_of_cells, 10] after selecting the ten
gene markers used in the experiment.
"""

from pathlib import Path

from aggregate_dynamics.data import load_npy_snapshot


DATA_DIR = Path("data/rna_seq")


def load_gene_snapshots(device=None):
    names = ["D0", "D2", "D4", "D7"]
    snapshots = {
        name: load_npy_snapshot(DATA_DIR / f"{name}.npy", device=device)
        for name in names
    }

    for name, x in snapshots.items():
        if x.ndim != 2 or x.shape[1] != 10:
            raise ValueError(
                f"{name} must have shape [n_cells, 10], got {tuple(x.shape)}"
            )

    return snapshots


if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(
            "Place locally authorized D0/D2/D4/D7 arrays under data/rna_seq/."
        )
    else:
        snapshots = load_gene_snapshots()
        for name, x in snapshots.items():
            print(name, tuple(x.shape))
