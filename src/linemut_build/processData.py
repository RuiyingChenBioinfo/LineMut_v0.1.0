import pandas as pd


def combine_ratio_depth(ad_ratio, ad_depth):
    """
    Combine SNV ratio (ad_ratio) and depth (ad_depth) into a single AnnData.

    Parameters
    ----------
    ad_ratio : AnnData
        SNV ratio data. Values must be in [0, 1].
    ad_depth : AnnData
        Depth data. Must cover the same obs/var as ad_ratio.

    Returns
    -------
    AnnData
        Copy of ad_ratio with depth stored in layers["depth"].
    """
    import numpy as np
    from scipy import sparse

    X = ad_ratio.X
    if sparse.issparse(X):
        data = X.data
    else:
        data = X

    if np.nanmin(data) < 0 or np.nanmax(data) > 1:
        raise ValueError("Ratio data should be in [0,1]")

    adata = ad_ratio.copy()
    r = ad_depth[adata.obs_names, adata.var_names]
    adata.layers["depth"] = r.X

    return adata


def filter_snv_by_depth_ratio(
    adata,
    min_tot_depth=20,
    min_mut_depth=0,
    min_mut_avg_ratio=0,
    x_layer=None,
    depth_layer="depth",
):
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    # ---- checks ----
    if depth_layer not in adata.layers:
        raise KeyError(f"Missing '{depth_layer}' in adata.layers.")
    depth = adata.layers[depth_layer]

    X = adata.layers[x_layer] if x_layer is not None else adata.X

    # depth could be per-cell (n_obs,) or (n_obs, 1), or per-cell-per-var (n_obs, n_vars)
    per_cell = (getattr(depth, "ndim", 2) == 1) or (hasattr(depth, "shape") and len(depth.shape) == 2 and depth.shape[1] == 1)

    # ---- total depth per SNV ----
    if per_cell:
        d = np.asarray(depth.toarray()).ravel() if sp.issparse(depth) else np.asarray(depth).ravel()
        denom_scalar = float(d.sum())
        # treat D_ij = d_i for all j (consistent with weighted_per_var per_cell branch)
        snv_depth_sums = pd.Series(np.full(adata.n_vars, denom_scalar, dtype=float), index=adata.var_names)
    else:
        if sp.issparse(depth):
            snv_depth_arr = np.asarray(depth.sum(axis=0)).ravel()
        else:
            snv_depth_arr = np.asarray(depth).sum(axis=0)

        if snv_depth_arr.shape[0] != adata.n_vars:
            raise ValueError("Depth matrix's number of columns does not match adata.var_names.")

        snv_depth_sums = pd.Series(snv_depth_arr.astype(float, copy=False), index=adata.var_names)

    # ---- mutated reads sums: sum_i X_ij * D_ij  (only compute if needed for min_mut_depth or min_mut_avg_ratio) ----
    need_num = (float(min_mut_depth) > 0) or (float(min_mut_avg_ratio) > 0)

    if need_num:
        if per_cell:
            d = np.asarray(depth.toarray()).ravel() if sp.issparse(depth) else np.asarray(depth).ravel()
            if sp.issparse(X):
                num = np.asarray(X.multiply(d[:, None]).sum(axis=0)).ravel()
            else:
                num = (np.asarray(X) * d[:, None]).sum(axis=0)
            num = num.astype(float, copy=False)
        else:
            if sp.issparse(X) and sp.issparse(depth):
                num = np.asarray(X.multiply(depth).sum(axis=0)).ravel()
            elif sp.issparse(X) and not sp.issparse(depth):
                # sparse X elementwise-multiply dense depth is supported
                num = np.asarray(X.multiply(np.asarray(depth)).sum(axis=0)).ravel()
            elif (not sp.issparse(X)) and sp.issparse(depth):
                num = np.asarray(depth.multiply(np.asarray(X)).sum(axis=0)).ravel()
            else:
                num = (np.asarray(X) * np.asarray(depth)).sum(axis=0)
            num = num.astype(float, copy=False)

        if num.shape[0] != adata.n_vars:
            raise ValueError("Mutated reads matrix's number of columns does not match adata.var_names.")

        snv_mut_sums = pd.Series(num, index=adata.var_names)
    else:
        snv_mut_sums = pd.Series(np.zeros(adata.n_vars, dtype=float), index=adata.var_names)

    # ---- weighted average ratio per SNV (same as weighted_per_var) ----
    if float(min_mut_avg_ratio) > 0:
        if per_cell:
            denom = float(snv_depth_sums.iloc[0])  # scalar
            mut_avg_ratio = np.divide(
                snv_mut_sums.to_numpy(),
                denom,
                out=np.full(adata.n_vars, np.nan, dtype=float),
                where=(denom != 0),
            )
        else:
            denom = snv_depth_sums.to_numpy()
            mut_avg_ratio = np.divide(
                snv_mut_sums.to_numpy(),
                denom,
                out=np.full(adata.n_vars, np.nan, dtype=float),
                where=(denom != 0),
            )
        mut_avg_ratio = pd.Series(mut_avg_ratio, index=adata.var_names)
    else:
        # not used in mask
        mut_avg_ratio = None

    # ---- build mask ----
    mask = (snv_depth_sums >= float(min_tot_depth))

    if float(min_mut_depth) > 0:
        mask &= (snv_mut_sums >= float(min_mut_depth))

    if float(min_mut_avg_ratio) > 0:
        mask &= (mut_avg_ratio >= float(min_mut_avg_ratio))

    snv_to_keep = snv_depth_sums.index[mask]

    print(f"SNV number before filter: {adata.n_vars}")
    print(f"SNV number after filter: {len(snv_to_keep)}")

    return adata[:, snv_to_keep].copy()


def test_all_differential_snv_by_cmb(
    adata,
    min_depth: float = 20,
    depth_layer: str = "depth",
    alternative: str = "two-sided",
    fdr_method: str = "fdr_bh",
    sort_by: str = "FDR",
    return_extra_cols: bool = True,
    fdr_threshold: float | None = 0.05,
    test: str = "fisher",
    chi2_correction: bool = True,
    ratio_diff_threshold: float = 0.2,
) -> pd.DataFrame:

    import numpy as np
    import scipy.sparse as sp
    import warnings
    from scipy.stats import fisher_exact, chi2_contingency
    from statsmodels.stats.multitest import multipletests

    test = str(test).lower()
    valid_tests = {"fisher", "chi2"}
    if test not in valid_tests:
        raise ValueError(f"Invalid test='{test}'. Must be one of {sorted(valid_tests)}.")

    if depth_layer not in adata.layers:
        raise KeyError(f"Missing layer in adata.layers: '{depth_layer}'")

    if test == "chi2" and (alternative is not None and str(alternative).lower() != "two-sided"):
        warnings.warn("Chi-square test ignores 'alternative'; it is effectively two-sided.")

    X = adata.X
    D = adata.layers[depth_layer]

    n_cmb, n_snv = adata.shape

    def _col_as_array(mat, col_index: int) -> np.ndarray:
        """Return a dense 1D numpy array for the specified column of a dense/sparse matrix."""
        if sp.issparse(mat):
            return mat[:, col_index].toarray().ravel()
        arr = np.asarray(mat)
        return arr[:, col_index].ravel()

    records = []

    for j in range(n_snv):
        snv_name = adata.var_names[j]

        depth_col = _col_as_array(D, j).astype(float)
        val_col = _col_as_array(X, j).astype(float)

        ratio = np.clip(val_col, 0.0, 1.0)
        valid = np.isfinite(ratio) & np.isfinite(depth_col) & (depth_col >= float(min_depth))

        if valid.sum() < 2:
            continue

        mut_counts_all = np.rint(ratio * depth_col).astype(int)
        mut_counts_all = np.clip(mut_counts_all, 0, depth_col.astype(int))
        ref_counts_all = depth_col.astype(int) - mut_counts_all

        total_mut = mut_counts_all[valid].sum()
        total_ref = ref_counts_all[valid].sum()

        valid_idx = np.where(valid)[0]
        for i in valid_idx:
            cmb_name = adata.obs_names[i]

            mut_i = int(mut_counts_all[i])
            ref_i = int(ref_counts_all[i])

            other_mut = int(total_mut - mut_i)
            other_ref = int(total_ref - ref_i)
            other_depth = int(other_mut + other_ref)
            other_ratio = other_mut / max(other_depth, 1) if other_depth > 0 else np.nan

            if not (np.isfinite(other_ratio) and abs(float(ratio[i]) - float(other_ratio)) >= float(ratio_diff_threshold)):
                continue

            if (mut_i + ref_i) == 0 or (other_mut + other_ref) == 0:
                p = np.nan
            else:
                table = [[mut_i, ref_i], [other_mut, other_ref]]
                try:
                    if test == "fisher":
                        _, p = fisher_exact(table, alternative=alternative)
                    else:
                        _, p, _, _ = chi2_contingency(table, correction=chi2_correction)
                except Exception:
                    p = np.nan

            rec = {
                "SNV": snv_name,
                "CMB": cmb_name,
                "depth": int(depth_col[i]) if np.isfinite(depth_col[i]) else np.nan,
                "mut_ratio": float(ratio[i]) if np.isfinite(ratio[i]) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            }

            if return_extra_cols:
                rec.update({
                    "other_depth": other_depth,
                    "other_mut_ratio": other_ratio,
                })

            records.append(rec)

    desired_order = ["SNV", "CMB", "depth", "other_depth", "mut_ratio", "other_mut_ratio", "p_value", "FDR"]
    if not records:
        if return_extra_cols:
            return pd.DataFrame(columns=desired_order)
        else:
            return pd.DataFrame(columns=[c for c in desired_order if c not in {"other_depth", "other_mut_ratio"}])

    df = pd.DataFrame.from_records(records)

    pvals = df["p_value"].values
    mask = np.isfinite(pvals)
    if mask.any():
        _, qvals, _, _ = multipletests(pvals[mask], method=fdr_method)
        df.loc[mask, "FDR"] = qvals
    else:
        df["FDR"] = np.nan

    if fdr_threshold is not None:
        df = df[df["FDR"].notna() & (df["FDR"] <= float(fdr_threshold))]

    key = "FDR" if str(sort_by).lower() == "fdr" else "p_value"
    df = df.sort_values(by=[key, "p_value"], na_position="last").reset_index(drop=True)

    present = [c for c in desired_order if c in df.columns]
    rest = [c for c in df.columns if c not in present]
    df = df[present + rest]

    return df


def test_comp_differential_snv_by_cmb(
    adata,
    CMB1: str,
    CMB2: str,
    min_depth: float = 20,
    depth_layer: str = "depth",
    alternative: str = "two-sided",
    fdr_method: str = "fdr_bh",
    sort_by: str = "FDR",
    return_extra_cols: bool = True,
    fdr_threshold: float | None = 0.1,
    test: str = "fisher",
    chi2_correction: bool = True,
    ratio_diff_threshold: float = 0.2,
) -> pd.DataFrame:

    import numpy as np
    import scipy.sparse as sp
    import warnings
    from scipy.stats import fisher_exact, chi2_contingency
    from statsmodels.stats.multitest import multipletests

    test = str(test).lower()
    valid_tests = {"fisher", "chi2"}
    if test not in valid_tests:
        raise ValueError(f"Invalid test='{test}'. Must be one of {sorted(valid_tests)}.")

    if depth_layer not in adata.layers:
        raise KeyError(f"Missing layer in adata.layers: '{depth_layer}'")

    if test == "chi2" and (alternative is not None and str(alternative).lower() != "two-sided"):
        warnings.warn("Chi-square test ignores 'alternative'; it is effectively two-sided.")

    if CMB1 == CMB2:
        raise ValueError("CMB1 and CMB2 must be different.")

    try:
        i1 = adata.obs_names.get_loc(CMB1)
    except Exception as e:
        raise KeyError(f"CMB1 '{CMB1}' not found in adata.obs_names.") from e
    try:
        i2 = adata.obs_names.get_loc(CMB2)
    except Exception as e:
        raise KeyError(f"CMB2 '{CMB2}' not found in adata.obs_names.") from e

    X = adata.X
    D = adata.layers[depth_layer]

    n_cmb, n_snv = adata.shape
    if n_cmb <= max(i1, i2):
        raise ValueError("CMB indices out of bounds for adata.X.")

    def _col_as_array(mat, col_index: int) -> np.ndarray:
        if sp.issparse(mat):
            return mat[:, col_index].toarray().ravel()
        arr = np.asarray(mat)
        return arr[:, col_index].ravel()

    records = []

    for j in range(n_snv):
        snv_name = adata.var_names[j]

        depth_col = _col_as_array(D, j).astype(float)
        val_col = _col_as_array(X, j).astype(float)

        ratio_col = np.clip(val_col, 0.0, 1.0)

        d1, d2 = depth_col[i1], depth_col[i2]
        r1, r2 = ratio_col[i1], ratio_col[i2]

        v1 = (np.isfinite(d1) and np.isfinite(r1) and d1 >= float(min_depth))
        v2 = (np.isfinite(d2) and np.isfinite(r2) and d2 >= float(min_depth))
        if not (v1 and v2):
            continue

        if abs(float(r1) - float(r2)) < float(ratio_diff_threshold):
            continue

        mut1 = int(np.clip(np.rint(r1 * d1), 0, int(d1)))
        ref1 = int(d1) - mut1
        mut2 = int(np.clip(np.rint(r2 * d2), 0, int(d2)))
        ref2 = int(d2) - mut2

        if (mut1 + ref1) == 0 or (mut2 + ref2) == 0:
            p = np.nan
        else:
            table = [[mut1, ref1], [mut2, ref2]]
            try:
                if test == "fisher":
                    _, p = fisher_exact(table, alternative=alternative)
                else:
                    _, p, _, _ = chi2_contingency(table, correction=chi2_correction)
            except Exception:
                p = np.nan

        rec = {
            "SNV": snv_name,
            "CMB1": CMB1,
            "CMB2": CMB2,
            "depth1": int(d1),
            "depth2": int(d2),
            "mut_ratio1": float(r1),
            "mut_ratio2": float(r2),
            "p_value": float(p) if np.isfinite(p) else np.nan,
        }

        if return_extra_cols:
            rec.update({
                "mut_count1": mut1,
                "ref_count1": ref1,
                "mut_count2": mut2,
                "ref_count2": ref2,
            })

        records.append(rec)

    desired_order = [
        "SNV", "CMB1", "CMB2", "depth1", "depth2",
        "mut_ratio1", "mut_ratio2", "p_value", "FDR"
    ]

    if not records:
        return pd.DataFrame(columns=desired_order)

    df = pd.DataFrame.from_records(records)

    pvals = df["p_value"].values
    mask = np.isfinite(pvals)
    if mask.any():
        _, qvals, _, _ = multipletests(pvals[mask], method=fdr_method)
        df.loc[mask, "FDR"] = qvals
    else:
        df["FDR"] = np.nan

    if fdr_threshold is not None:
        df = df[df["FDR"].notna() & (df["FDR"] <= float(fdr_threshold))]

    key = "FDR" if str(sort_by).lower() == "fdr" else "p_value"
    df = df.sort_values(by=[key, "p_value"], na_position="last").reset_index(drop=True)

    df = df[desired_order]

    return df


def add_betweenness_centrality_to_adata(
    adata_joint,
    layer="joint_connectivities_sym",
    weight_transform="inv",
    quantile_thr=0.75,
    eps=1e-12,
    add_obs_name="between_centrality",
    plot_bet_centrality_box=True,
    box_figsize=(7, 3),
    annotate_high=True,
    jitter=0.06,
    random_state=0,
    savepath=None,
    dpi=300,
    transparent=False,
    bbox_inches="tight",
):
    import numpy as np
    import scipy.sparse as sp

    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx is required: pip install networkx") from e

    import matplotlib.pyplot as plt

    if not isinstance(add_obs_name, str) or len(add_obs_name.strip()) == 0:
        raise ValueError("add_obs_name must be a non-empty string.")
    add_obs_name = add_obs_name.strip()
    add_obs_name_high = f"{add_obs_name}_high"

    if layer not in adata_joint.obsp:
        raise KeyError(
            f"'{layer}' not found in adata_joint.obsp. Available keys: {list(adata_joint.obsp.keys())}"
        )

    A = adata_joint.obsp[layer]
    W = A.tocsr() if sp.issparse(A) else sp.csr_matrix(np.asarray(A))

    if W.shape[0] != W.shape[1] or W.shape[0] != adata_joint.n_obs:
        raise ValueError(f"Matrix shape {W.shape} does not match adata_joint.n_obs={adata_joint.n_obs}.")

    if weight_transform not in {"inv", "neglog"}:
        raise ValueError("weight_transform must be 'inv' or 'neglog'.")

    labels = adata_joint.obs_names.astype(str).to_numpy()
    n = len(labels)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    W = W.tocsr()
    W.setdiag(0.0)
    W.eliminate_zeros()

    Wu = sp.triu(W, k=1).tocoo()
    for i, j, w in zip(Wu.row, Wu.col, Wu.data):
        w = float(w)
        if w <= 0.0:
            continue

        if weight_transform == "inv":
            length = 1.0 / (w + eps)
        else:
            ww = float(np.clip(w, eps, 1.0))
            length = -float(np.log(ww))

        G.add_edge(int(i), int(j), length=float(length), sim=float(w))

    bc = nx.betweenness_centrality(G, weight="length", normalized=True)
    bet = np.array([bc.get(i, 0.0) for i in range(n)], dtype=float)

    adata_joint.obs[add_obs_name] = bet

    thr = float(np.quantile(bet, quantile_thr)) if n > 0 else 0.0
    high = bet > thr
    adata_joint.obs[add_obs_name_high] = np.where(high, "Yes", "No")

    q1 = float(np.quantile(bet, 0.25)) if n > 0 else 0.0
    med = float(np.quantile(bet, 0.50)) if n > 0 else 0.0
    q3 = float(np.quantile(bet, 0.75)) if n > 0 else 0.0

    fig_ax = None
    if plot_bet_centrality_box:
        rng = np.random.default_rng(random_state)
        y = 1.0 + rng.uniform(-jitter, jitter, size=n)

        fig, ax = plt.subplots(figsize=box_figsize)
        ax.boxplot(bet, vert=False)
        ax.scatter(bet, y, s=18, alpha=0.7)

        if annotate_high:
            for i in np.where(high)[0]:
                ax.text(
                    float(bet[i]),
                    float(y[i]) + 0.03,
                    str(labels[i]),
                    fontsize=8,
                    rotation=45,
                    ha="left",
                    va="bottom",
                )

        ax.set_yticks([])
        ax.set_xlabel("Betweenness centrality")
        ax.set_title(f"Betweenness centrality | layer={layer} | transform={weight_transform} | obs={add_obs_name}")
        plt.tight_layout()
        fig_ax = (fig, ax)

        if savepath is not None:
            fig.savefig(
                savepath,
                dpi=dpi,
                bbox_inches=bbox_inches,
                transparent=transparent,
            )

    return {
        "layer_used": layer,
        "add_obs_name": add_obs_name,
        "add_obs_name_high": add_obs_name_high,
        "threshold_quantile": float(quantile_thr),
        "threshold_value": thr,
        "n_high": int(high.sum()),
        "summary_three_points": {"Q1": q1, "Median": med, "Q3": q3},
        "high_labels": labels[high].tolist(),
        "plot": fig_ax,
        "savepath": savepath,
    }


def spatial_filter_conn_by_dist(
    adata,
    conn_key="joint_connectivities_sym",
    x_key="x_center",
    y_key="y_center",
    new_key="joint_connectivities_sym_filteredbyDist",
    dist_quantile=0.95,
    min_keep_per_node=2,
    symmetrize_input=False,
    zero_diagonal=True,
    return_stats=True,
):
    """
    Mask long range edges in a connectivity matrix using physical distances.

    Steps
    1) Use only existing edges (W > 0) to form the edge distance distribution.
    2) Identify long distance edges by quantile threshold: r = quantile(D_edges, dist_quantile).
    3) Set edges with D > r to zero, but enforce a minimum connectivity constraint:
       for each node, keep its min_keep_per_node nearest (by physical distance) original neighbors.
    """
    import numpy as np
    import scipy.sparse as sp

    if conn_key not in adata.obsp:
        raise KeyError(f"{conn_key!r} not found in adata.obsp. Available: {list(adata.obsp.keys())}")

    for k in (x_key, y_key):
        if k not in adata.obs.columns:
            raise KeyError(f"{k!r} not found in adata.obs.columns")

    if not (0.0 < float(dist_quantile) < 1.0):
        raise ValueError("dist_quantile must be in (0, 1). Example: 0.95")

    x = adata.obs[x_key].to_numpy()
    y = adata.obs[y_key].to_numpy()
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        bad = np.where((~np.isfinite(x)) | (~np.isfinite(y)))[0]
        raise ValueError(
            f"Found non finite {x_key}/{y_key} for {len(bad)} rows in adata.obs. "
            f"Please fill them before masking."
        )

    A = adata.obsp[conn_key]
    if sp.issparse(A):
        W = A.tocsr(copy=True)
    else:
        W = sp.csr_matrix(np.asarray(A, dtype=float))

    n = adata.n_obs
    if W.shape != (n, n):
        raise ValueError(f"{conn_key!r} has shape {W.shape}, but adata.n_obs is {n}")

    W.eliminate_zeros()

    if symmetrize_input:
        W = (W + W.T) * 0.5
        W = W.tocsr()
        W.eliminate_zeros()

    if zero_diagonal:
        W.setdiag(0.0)
        W.eliminate_zeros()

    if new_key is None:
        new_key = f"{conn_key}_spatialmasked"

    coo = W.tocoo()
    mask_upper = coo.row < coo.col
    rows = coo.row[mask_upper].astype(int, copy=False)
    cols = coo.col[mask_upper].astype(int, copy=False)
    vals = coo.data[mask_upper].astype(float, copy=False)

    n_edges = rows.size
    if n_edges == 0:
        W_new = W.copy()
        adata.obsp[new_key] = W_new
        stats = {
            "conn_key_in": conn_key,
            "conn_key_out": new_key,
            "n_nodes": int(n),
            "n_edges_input_undirected": 0,
            "dist_quantile": float(dist_quantile),
            "threshold_r": None,
            "min_keep_per_node": int(min_keep_per_node),
            "n_edges_removed": 0,
            "n_edges_output_undirected": 0,
        }
        return (W_new, stats) if return_stats else W_new

    dx = x[rows] - x[cols]
    dy = y[rows] - y[cols]
    dist = np.sqrt(dx * dx + dy * dy)

    r = float(np.quantile(dist, float(dist_quantile)))
    outlier = dist > r

    protected = set()
    m = int(min_keep_per_node)
    if m > 0:
        for i in range(n):
            row_i = W.getrow(i)
            nbrs = row_i.indices
            if nbrs.size == 0:
                continue

            dxi = x[nbrs] - x[i]
            dyi = y[nbrs] - y[i]
            di = np.sqrt(dxi * dxi + dyi * dyi)

            if nbrs.size <= m:
                keep_nbrs = nbrs
            else:
                keep_nbrs = nbrs[np.argsort(di)[:m]]

            for j in keep_nbrs:
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                if a != b:
                    protected.add((a, b))

    keep = np.ones(n_edges, dtype=bool)
    if np.any(outlier):
        for k_idx in np.where(outlier)[0]:
            a = int(rows[k_idx])
            b = int(cols[k_idx])
            if (a, b) not in protected:
                keep[k_idx] = False

    rows_k = rows[keep]
    cols_k = cols[keep]
    vals_k = vals[keep]

    W_new = sp.coo_matrix((vals_k, (rows_k, cols_k)), shape=(n, n)).tocsr()
    W_new = W_new + W_new.T
    if zero_diagonal:
        W_new.setdiag(0.0)
    W_new.eliminate_zeros()

    adata.obsp[new_key] = W_new

    stats = {
        "conn_key_in": conn_key,
        "conn_key_out": new_key,
        "n_nodes": int(n),
        "n_edges_input_undirected": int(n_edges),
        "dist_quantile": float(dist_quantile),
        "threshold_r": r,
        "min_keep_per_node": int(min_keep_per_node),
        "n_outlier_edges": int(outlier.sum()),
        "n_edges_removed": int((~keep).sum()),
        "n_edges_output_undirected": int(vals_k.size),
        "dist_min": float(dist.min()) if dist.size else None,
        "dist_median": float(np.median(dist)) if dist.size else None,
        "dist_max": float(dist.max()) if dist.size else None,
    }

    return (W_new, stats) if return_stats else W_new


def ternary_cmb_snv(
    adata,
    rare_max_occ=3,
    mid_max_occ=13,
    title=None,
    point_size=80,
    alpha=0.9,
    show_labels=False,
    label_fontsize=8,
    label_subset=None,
    label_jitter=0.008,
    random_state=0,
    minmax_each=False,
    group_colors=None,
    color_by=None,
    fallback_cmap="tab20",
    rank_obs_name="cmb_snv_ratio_rank",
    rank_tie_method="mergesort",
    figsize=(7, 7),
    savepath=None,
    dpi=300,
    show=True,
    p_rare_obs="p_rare",
    p_mid_obs="p_mid",
    p_common_obs="p_common",
):
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import matplotlib.pyplot as plt

    X = adata.X
    labels = adata.obs_names.astype(str).to_numpy()
    n = adata.n_obs
    if n == 0:
        raise ValueError("adata has no observations")

    if color_by is not None and color_by not in adata.obs.columns:
        raise KeyError(f"{color_by!r} not found in adata.obs.columns")

    if sp.issparse(X):
        X_bin = (X > 0).astype(np.int8).tocsr()
        feat_counts = np.asarray(X_bin.sum(axis=0)).ravel()
    else:
        X_bin = (np.asarray(X) > 0).astype(np.int8)
        feat_counts = X_bin.sum(axis=0)

    rare_mask = (feat_counts >= 1) & (feat_counts <= int(rare_max_occ))
    mid_mask = (feat_counts > int(rare_max_occ)) & (feat_counts <= int(mid_max_occ))
    common_mask = feat_counts > int(mid_max_occ)

    if sp.issparse(X_bin):
        present_count = np.asarray(X_bin.sum(axis=1)).ravel().astype(float)
        rare_count = np.asarray(X_bin[:, rare_mask].sum(axis=1)).ravel().astype(float)
        mid_count = np.asarray(X_bin[:, mid_mask].sum(axis=1)).ravel().astype(float)
        common_count = np.asarray(X_bin[:, common_mask].sum(axis=1)).ravel().astype(float)
    else:
        present_count = X_bin.sum(axis=1).astype(float)
        rare_count = X_bin[:, rare_mask].sum(axis=1).astype(float)
        mid_count = X_bin[:, mid_mask].sum(axis=1).astype(float)
        common_count = X_bin[:, common_mask].sum(axis=1).astype(float)

    eps = 1e-12
    denom = np.maximum(present_count, eps)
    p_rare = rare_count / denom
    p_mid = mid_count / denom
    p_common = common_count / denom

    s = p_rare + p_mid + p_common
    ok = s > 0
    p_rare[ok] = p_rare[ok] / s[ok]
    p_mid[ok] = p_mid[ok] / s[ok]
    p_common[ok] = p_common[ok] / s[ok]

    if minmax_each:
        def _minmax(a):
            a2 = a.copy()
            lo = np.nanmin(a2)
            hi = np.nanmax(a2)
            if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
                return np.zeros_like(a2)
            return (a2 - lo) / (hi - lo)

        a = _minmax(p_rare)
        b = _minmax(p_mid)
        c = _minmax(p_common)
        ss = a + b + c
        nz = ss > 0
        a[nz] = a[nz] / ss[nz]
        b[nz] = b[nz] / ss[nz]
        c[nz] = c[nz] / ss[nz]
        a[~nz] = 1.0 / 3.0
        b[~nz] = 1.0 / 3.0
        c[~nz] = 1.0 / 3.0
        p_rare, p_mid, p_common = a, b, c

    adata.obs[p_rare_obs] = p_rare
    adata.obs[p_mid_obs] = p_mid
    adata.obs[p_common_obs] = p_common

    score = p_common - p_rare
    order = np.argsort(-score, kind=str(rank_tie_method))
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adata.obs[rank_obs_name] = ranks

    h = np.sqrt(3) / 2.0
    xs = 0.5 * p_rare + p_mid
    ys = h * p_rare

    if color_by is None:
        group_vals = labels
    else:
        group_vals = adata.obs[color_by].astype(str).to_numpy()

    if group_colors is None:
        cm = plt.cm.get_cmap(fallback_cmap, max(n, 1))
        face_colors = [cm(i) for i in range(n)]
    else:
        default_cm = plt.cm.get_cmap(fallback_cmap, max(len(np.unique(group_vals)), 1))
        uniq = list(dict.fromkeys(group_vals.tolist()))
        default_map = {g: default_cm(i) for i, g in enumerate(uniq)}
        face_colors = []
        for g in group_vals:
            if g in group_colors:
                face_colors.append(group_colors[g])
            else:
                face_colors.append(default_map.get(g, "#cccccc"))

    fig, ax = plt.subplots(figsize=figsize)

    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, h, 0.0]
    ax.plot(tri_x, tri_y, linewidth=2, color="#000000")

    ax.scatter(
        xs, ys,
        s=point_size,
        c=face_colors,
        alpha=alpha,
        edgecolors="#000000",
        linewidths=1.0,
        zorder=3,
    )

    ax.text(0.5, h + 0.04, f"Rare (occ <= {rare_max_occ})", ha="center", va="bottom")
    ax.text(-0.02, -0.03, f"Common (> {mid_max_occ})", ha="left", va="top")
    ax.text(1.02, -0.03, f"Mid ({rare_max_occ + 1}..{mid_max_occ})", ha="right", va="top")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, h + 0.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal", adjustable="box")

    if title is None:
        title = f"Rare vs Common vs Mid | Rare<= {rare_max_occ}, Mid<= {mid_max_occ}"
        if minmax_each:
            title += " | minmax_each=True"
    ax.set_title(title)

    if show_labels:
        if label_subset is None:
            keep_idx = np.arange(n)
        else:
            if isinstance(label_subset, (list, tuple, set, np.ndarray, pd.Series)):
                keep_set = set([str(x) for x in label_subset])
            else:
                keep_set = {str(label_subset)}
            keep_idx = np.array([i for i, lb in enumerate(labels) if lb in keep_set], dtype=int)

        texts = []
        for i in keep_idx:
            texts.append(
                ax.text(xs[i], ys[i], labels[i], fontsize=label_fontsize,
                        ha="center", va="center", color="#000000", zorder=5)
            )

        try:
            from adjustText import adjust_text
            adjust_text(
                texts,
                ax=ax,
                expand_points=(1.2, 1.4),
                expand_text=(1.2, 1.4),
                force_points=0.3,
                force_text=0.7,
                only_move={"points": "xy", "text": "xy"},
                arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.7, color="#000000"),
            )
        except ImportError:
            rng = np.random.default_rng(int(random_state))
            for t in texts:
                x0, y0 = t.get_position()
                dx, dy = rng.uniform(-label_jitter, label_jitter, size=2)
                t.set_position((x0 + dx, y0 + dy))

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return (fig, ax)