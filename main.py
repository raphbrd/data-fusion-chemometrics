import os.path as op
import numpy as np
import pandas as pd
import tensorly as tl
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl

from cmtf import CMTF
from utils import replace_nan, match_score, normalize_pos_factors
from visualization import plot_match_scores, plot_concentrations_per_mixtures


def compute_parafac(tensor, rank, non_negative=False):
    """ Compute the PARAFAC decomposition of a tensor.

    Parameters
    ----------
    tensor: np.ndarray
        The tensor to decompose.
    rank: int
        The rank of the decomposition.
    non_negative: bool
        If True, enforce non-negativity constraints on the factors.

    Returns
    -------
    tuple: (weight, factors)
        A tuple of the weights and normalized factors of the PARAFAC decomposition.
    """
    if non_negative:
        from tensorly.decomposition import non_negative_parafac as parafac
    else:
        from tensorly.decomposition import parafac

    factors = parafac(tensor, rank=rank)
    return tl.cp_normalize(factors)


def load_and_preprocess_data(data_path):
    concentrations = pd.read_table(op.join(data_path, "concentrations.txt"), sep="\s+")
    gt_concentrations = normalize_pos_factors(concentrations.values)
    mat = sio.loadmat(op.join(data_path, "EEM_NMR_LCMS.mat"))

    # load and preprocess the data
    eem = mat["X"][0][0]["data"]
    eem = replace_nan(eem, strategy="zero")
    nmr = mat["Y"][0][0]["data"]
    nmr = replace_nan(nmr, strategy="zero")
    lcms = mat["Z"][0][0]["data"]
    lcms = replace_nan(lcms, strategy="zero")

    return eem, nmr, lcms, gt_concentrations, concentrations


def main_parafac(data_path, fig_path, output_path, rank, max_iter):
    eem, nmr, lcms, gt_concentrations, concentrations = load_and_preprocess_data(data_path)

    print("Computing PARAFAC decomposition on each tensor")
    fac_eem = compute_parafac(eem, rank)
    fac_nmr = compute_parafac(nmr, rank)
    u1_eem, u1_nmr = fac_eem[1][0], fac_nmr[1][0]
    # change for PCA
    svd_lcms = np.linalg.svd(lcms, full_matrices=False)
    v1_lcms = svd_lcms[-1][:, :rank]
    print("PARAFAC done.")

    scores = []
    for factor in [u1_nmr, u1_eem, v1_lcms]:
        estimated_concentration = normalize_pos_factors(factor)
        u1_score = match_score(gt_concentrations, estimated_concentration)
        scores.append(u1_score)

    fig = plot_match_scores(
        scores,
        labels=["NMR", "EEM", "LCMS"],
        x_ticks=np.arange(5),
        x_tick_labels=concentrations.columns,
        colors=mpl.colormaps['Dark2'].colors[3:],
        show=False,
    )
    plt.gca().tick_params(labelrotation=45)
    fig.tight_layout()
    fig.savefig(op.join(fig_path, "one_way_data_decomposition_comparison.pdf"), dpi=300)
    plt.close(fig)


def main_cmtf(data_path, output_path, rank, max_iter):
    eem, nmr, lcms, gt_concentrations, concentrations = load_and_preprocess_data(data_path)

    # Compute CMTF for two-way data (one tensor and the matrix) and 3-way data (two tensors and one matrix)
    cmtf = CMTF(max_iter=max_iter)

    # EEM+LCMS
    out_eem = cmtf.fit([eem], [lcms], rank=rank)
    cmtf.save_fit(
        op.join(output_path, f"cmtf_eem_rank_{rank}_{max_iter}_iters.pkl"),
        out_eem,
        deltas_filename=op.join(output_path, f"cmtf_eem_rank_{rank}_{max_iter}_iters_deltas.pkl"),
    )

    # NMR+LCMS
    out_nmr = cmtf.fit([nmr], [lcms], rank=rank)
    cmtf.save_fit(
        op.join(output_path, f"cmtf_nmr_rank_{rank}_{max_iter}_iters.pkl"),
        out_nmr,
        deltas_filename=op.join(output_path, f"cmtf_nmr_rank_{rank}_{max_iter}_iters_deltas.pkl"),
    )

    # 3-way data
    out_3way = cmtf.fit([eem, nmr], [lcms], rank=rank)
    cmtf.save_fit(
        op.join(output_path, f"cmtf_3way_rank_{rank}_{max_iter}_iters.pkl"),
        out_3way,
        deltas_filename=op.join(output_path, f"cmtf_3way_rank_{rank}_{max_iter}_iters_deltas.pkl"),
    )


def plot_cmtf_results(data_path, fig_path, output_path, max_iter, rank):
    """ Plot the results of the CMTF decomposition for the Acar data."""
    concentrations = pd.read_table(op.join(data_path, "concentrations.txt"), sep="\s+")
    gt_concentrations = normalize_pos_factors(concentrations.values)

    cmft = CMTF(max_iter=max_iter)
    out_eem_lcms = cmft.load_fit(op.join(output_path, f"cmtf_eem_rank_{rank}_{max_iter}_iters.pkl"))
    out_nmr_lcms = cmft.load_fit(op.join(output_path, f"cmtf_nmr_rank_{rank}_{max_iter}_iters.pkl"))
    out_3way = cmft.load_fit(op.join(output_path, f"cmtf_3way_rank_{rank}_{max_iter}_iters.pkl"))

    # looking at the common factor across the three decompositions
    u1_cmtf_eem = out_eem_lcms[0]
    u1_cmtf_nmr = out_nmr_lcms[0]
    u1_cmtf_3way = out_3way[0]

    colors = mpl.colormaps['Dark2'].colors[:3]
    run_name = ["cmtf_eem", "cmtf_nmr", "cmtf_3way"]
    scores = []
    for run_idx, factor in enumerate([u1_cmtf_eem, u1_cmtf_nmr, u1_cmtf_3way]):
        estimated_concentration = normalize_pos_factors(factor)
        u1_score, _, col_inds = match_score(gt_concentrations, estimated_concentration, return_indices=True)
        scores.append(u1_score)

        fig = plot_concentrations_per_mixtures(
            gt_concentrations,
            estimated_concentration,
            col_inds,
            col_labels=concentrations.columns,
            color_gt="black",
            color_hat=colors[run_idx],
        )
        fig.savefig(op.join(fig_path, f"{run_name[run_idx]}_concentrations.pdf"), dpi=300)

    fig = plot_match_scores(
        scores,
        labels=["EEM+LCMS", "NMR+LCMS", "3-way"],
        x_ticks=np.arange(5),
        x_tick_labels=concentrations.columns,
        colors=colors,
        show=False,
    )
    plt.gca().tick_params(labelrotation=45)
    fig.tight_layout()
    fig.savefig(op.join(fig_path, f"two_way_vs_three_way_data_decomposition_comparison_rank_{rank}.pdf"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="CMTF for 3-way data")
    parser.add_argument("--data_path", type=str, default="./Acar_data", help="Path to the data file")
    parser.add_argument("--fig_path", type=str, default="./figures", help="Path to save the figures", )
    parser.add_argument("--output_path", type=str, default="./output_data", help="Path to save the output data")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the decomposition")
    parser.add_argument("--max-iter", type=int, required=True, help="Maximum number of iterations")
    parser.add_argument("--parafac", action="store_true", help="Compute the PARAFAC decomposition")
    parser.add_argument("--cmtf", action="store_true", help="Compute the CMTF decomposition")
    args = parser.parse_args()
    logging.info(f"Using data path:   {args.data_path}")
    logging.info(f"Using figure path: {args.fig_path}")
    logging.info(f"Using output path: {args.output_path}")

    if args.parafac:
        main_parafac(
            data_path=args.data_path,
            fig_path=args.fig_path,
            output_path=args.output_path,
            rank=args.rank,
            max_iter=args.max_iter
        )

    if args.cmtf:
        main_cmtf(
            data_path=args.data_path,
            output_path=args.output_path,
            rank=args.rank,
            max_iter=args.max_iter
        )

    if not args.cmtf and not args.parafac:
        logging.warning("No decomposition method specified. Use either --parafac or --cmtf.")

    plot_cmtf_results(
        data_path=args.data_path,
        fig_path=args.fig_path,
        output_path=args.output_path,
        max_iter=args.max_iter,
        rank=args.rank
    )
