"""Brain Twin Extractor — Phase 6.

Extracts the patient-specific brain state vector Z_t from a trained
temporal model at each clinical visit, enabling trajectory visualization
and formal Digital Brain Twin analysis.

The brain state Z_t is defined as:
    - For Clinical-only GRU: the final hidden state h_n at each visit
    - For Bimodal fusion:    [clinical_hidden ‖ mri_embedding]

This module loads trained model checkpoints and extracts Z_t trajectories
per patient across all their visits.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.nn.utils.rnn import pack_padded_sequence

logger = logging.getLogger(__name__)


class ClinicalBrainTwinExtractor:
    """Extracts Z_t trajectories from the trained clinical GRU.

    Z_t is defined as the GRU hidden state at each visit timestep,
    representing the model's learned brain state at that point in time.

    For a patient with visits [V1, V2, V3], we extract:
        Z_1 = GRU([V1])
        Z_2 = GRU([V1, V2])
        Z_3 = GRU([V1, V2, V3])

    This gives the temporal evolution of the brain state representation.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device | None = None):
        """Initialize the extractor.

        Args:
            model:  Trained GRU-based model with a `gru` attribute.
            device: Torch device (defaults to CPU).
        """
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()

        # Verify the model has a GRU component
        if not hasattr(self.model, "gru"):
            raise AttributeError(
                "Model must have a `gru` attribute (nn.GRU). "
                "The ClinicalBrainTwinExtractor works with TemporalCerebroNet."
            )

    @torch.no_grad()
    def extract_patient_trajectory(
        self,
        visit_features: np.ndarray,
    ) -> np.ndarray:
        """Extract Z_t at each timestep for a single patient.

        Args:
            visit_features: Array of shape (n_visits, num_features) —
                            ordered clinical features for each visit.

        Returns:
            trajectories: Array of shape (n_visits, hidden_dim) —
                          Z_t at each timestep.
        """
        n_visits, num_features = visit_features.shape
        trajectories = []

        for t in range(1, n_visits + 1):
            # Build prefix sequence: visits 0..t-1
            seq = torch.tensor(
                visit_features[:t], dtype=torch.float32
            ).unsqueeze(0).to(self.device)  # (1, t, features)
            length = torch.tensor([t], dtype=torch.long)

            # Pass through GRU
            packed = pack_padded_sequence(
                seq, length.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.model.gru(packed)
            z_t = h_n[-1].squeeze(0).cpu().numpy()  # (hidden_dim,)
            trajectories.append(z_t)

        return np.stack(trajectories, axis=0)  # (n_visits, hidden_dim)

    def extract_all_patients(
        self,
        dataset,
        pairs_df: pd.DataFrame,
        imputer,
        scaler,
    ) -> dict[str, np.ndarray]:
        """Extract Z_t trajectories for all patients.

        Args:
            dataset:   SequenceDataset (used to access feature names/preprocessing).
            pairs_df:  Longitudinal pairs DataFrame.
            imputer:   Fitted imputer from training dataset.
            scaler:    Fitted scaler from training dataset.

        Returns:
            Dict mapping subject_id → trajectory array (n_visits, hidden_dim).
        """
        from cerebro_x.features.clinical import build_feature_matrix

        X_df, _ = build_feature_matrix(pairs_df)
        X_raw = X_df.values
        X_imp = imputer.transform(X_raw)
        X_scaled = np.asarray(scaler.transform(X_imp))

        subject_ids = pairs_df["Subject ID"].values
        visit_nums = pairs_df["current_visit"].values

        # Group by subject, sort by visit
        from collections import defaultdict
        subject_visits: dict[str, list[tuple]] = defaultdict(list)
        for i, (sid, visit) in enumerate(zip(subject_ids, visit_nums)):
            subject_visits[sid].append((int(visit), X_scaled[i]))

        trajectories: dict[str, np.ndarray] = {}
        for sid, visit_list in subject_visits.items():
            visit_list.sort(key=lambda x: x[0])
            features = np.stack([v[1] for v in visit_list], axis=0)
            trajectories[sid] = self.extract_patient_trajectory(features)
            logger.info(
                "Extracted Z_t trajectory for %s: %d visits, shape %s",
                sid, len(visit_list), trajectories[sid].shape
            )

        return trajectories

    def save_trajectories(
        self,
        trajectories: dict[str, np.ndarray],
        output_dir: Path,
    ) -> None:
        """Save trajectories as .npy files, one per patient.

        Args:
            trajectories: Dict from extract_all_patients().
            output_dir:   Directory to save files into.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for sid, traj in trajectories.items():
            save_path = output_dir / f"{sid}_Z_trajectory.npy"
            np.save(save_path, traj)

        # Also save a combined array
        all_ids = list(trajectories.keys())
        meta_path = output_dir / "subject_list.txt"
        meta_path.write_text("\n".join(all_ids))

        logger.info("Saved %d patient trajectories to %s", len(trajectories), output_dir)

    def visualize_trajectories_pca(
        self,
        trajectories: dict[str, np.ndarray],
        pairs_df: pd.DataFrame,
        output_dir: Path,
        max_patients: int = 20,
    ) -> None:
        """PCA visualization of Z_t trajectories in 2D.

        Projects all brain state vectors across all patients into 2D PCA space
        and plots trajectories, coloring by CDR progression.

        Args:
            trajectories:  Dict from extract_all_patients().
            pairs_df:      Longitudinal pairs DataFrame (for CDR ground truth).
            output_dir:    Directory to save plots.
            max_patients:  Maximum number of patients to plot (for readability).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Gather all Z_t vectors for PCA fitting
        all_z = []
        all_labels = []
        subject_indices = {}  # sid → list of indices into all_z

        cdr_map = pairs_df.groupby("Subject ID")["next_CDR"].apply(list).to_dict()

        patient_list = list(trajectories.keys())[:max_patients]

        for sid in patient_list:
            traj = trajectories[sid]
            start_idx = len(all_z)
            for z in traj:
                all_z.append(z)
            subject_indices[sid] = list(range(start_idx, len(all_z)))

            # CDR labels for color: use next_CDR values
            cdrs = cdr_map.get(sid, [])
            # Extend to trajectory length with last known value
            while len(cdrs) < len(traj):
                cdrs.append(cdrs[-1] if cdrs else 0.0)
            all_labels.extend(cdrs[:len(traj)])

        if len(all_z) < 2:
            logger.warning("Not enough Z_t vectors for PCA. Skipping visualization.")
            return

        all_z_array = np.stack(all_z, axis=0)
        n_components = min(2, all_z_array.shape[1], all_z_array.shape[0])
        pca = PCA(n_components=n_components)
        z_2d = pca.fit_transform(all_z_array)

        explained = pca.explained_variance_ratio_
        logger.info(
            "PCA variance explained: PC1=%.1f%%, PC2=%.1f%%",
            explained[0] * 100, explained[1] * 100 if len(explained) > 1 else 0
        )

        # Plot
        cdr_to_color = {0.0: "#2196F3", 0.5: "#FF9800", 1.0: "#F44336", 2.0: "#9C27B0"}
        cdr_labels = {0.0: "CDR 0.0 (Normal)", 0.5: "CDR 0.5 (MCI)", 1.0: "CDR 1.0 (Mild AD)", 2.0: "CDR 2.0 (Moderate AD)"}

        fig, ax = plt.subplots(figsize=(12, 8))

        for sid in patient_list:
            indices = subject_indices[sid]
            if len(indices) < 1:
                continue
            z_patient = z_2d[indices]
            cdrs_patient = [all_labels[i] for i in indices]

            # Draw trajectory line
            if len(z_patient) > 1:
                ax.plot(
                    z_patient[:, 0], z_patient[:, 1],
                    color="gray", alpha=0.3, linewidth=0.8, zorder=1
                )

            # Draw points colored by CDR
            for j, (z, cdr) in enumerate(zip(z_patient, cdrs_patient)):
                color = cdr_to_color.get(float(cdr), "black")
                marker = "o" if j == 0 else ("s" if j == len(z_patient) - 1 else "^")
                ax.scatter(z[0], z[1], c=color, s=60, marker=marker, zorder=2, alpha=0.8)

        # Legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor=c, label=lbl)
            for cdr, (c, lbl) in zip(
                cdr_to_color.keys(),
                zip(cdr_to_color.values(), cdr_labels.values())
            )
        ] + [
            Line2D([0], [0], marker="o", color="gray", label="First visit", markersize=8),
            Line2D([0], [0], marker="s", color="gray", label="Last visit", markersize=8),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)", fontsize=11)
        ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)" if len(explained) > 1 else "PC2", fontsize=11)
        ax.set_title(
            "Digital Brain Twin — Z_t Trajectories (PCA)\n"
            "Each line = one patient's brain state evolution across visits",
            fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        plot_path = output_dir / "brain_twin_Z_trajectories_PCA.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("PCA trajectory plot saved: %s", plot_path)

    def visualize_progression_heatmap(
        self,
        trajectories: dict[str, np.ndarray],
        output_dir: Path,
        max_patients: int = 30,
        max_visits: int = 5,
    ) -> None:
        """Heatmap of Z_t hidden dimensions across visits per patient.

        Args:
            trajectories:  Dict from extract_all_patients().
            output_dir:    Directory to save plots.
            max_patients:  Patients to display.
            max_visits:    Maximum visits per patient to show.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        patient_list = list(trajectories.keys())[:max_patients]

        # For each patient, take mean Z_t per visit, plot first 16 dims
        n_dims_to_show = min(16, list(trajectories.values())[0].shape[1])

        fig, axes = plt.subplots(
            len(patient_list), 1,
            figsize=(12, max(6, len(patient_list) * 0.8)),
            squeeze=False
        )

        for row_idx, sid in enumerate(patient_list):
            traj = trajectories[sid][:max_visits, :n_dims_to_show]
            ax = axes[row_idx, 0]
            im = ax.imshow(traj.T, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
            ax.set_ylabel(sid, fontsize=6, rotation=0, labelpad=60)
            ax.set_yticks([])
            ax.set_xticks(range(traj.shape[0]))
            ax.set_xticklabels([f"V{i+1}" for i in range(traj.shape[0])], fontsize=7)

        fig.suptitle("Brain Twin Z_t Heatmap — Hidden Dimensions Across Visits", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=axes[:, 0], shrink=0.6, label="Activation")

        plot_path = output_dir / "brain_twin_Z_heatmap.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Heatmap saved: %s", plot_path)
