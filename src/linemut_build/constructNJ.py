def revise_snv_ratio(
    adata,
    layer_name="original_ratio",
    low=0.1,
    high=0.9,
    mid=0.5,
    copy=True,
    dtype="float32",
):
    """
    1) Copy adata.X into a new layer (default: "original_ratio").
    2) Discretize adata.X in-place (or on a copy) using configurable thresholds:
         - values < low   -> 0
         - values > high  -> 1
         - otherwise      -> mid
       Note: values exactly equal to low/high fall into the "otherwise" bucket.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    layer_name : str
        Name of the layer that will store the original (pre-discretization) X.
    low : float
        Lower threshold.
    high : float
        Upper threshold.
    mid : float
        Value assigned to entries within [low, high] (inclusive).
    copy : bool
        If True, operate on a copy and return it; if False, modify the input adata in-place.
    dtype : str or numpy dtype
        Output dtype for dense X and the stored original layer (dense). Sparse keeps its dtype.

    Returns
    -------
    AnnData
        AnnData containing layers[layer_name] = original X, and modified X.
    """
    import numpy as np
    from scipy import sparse

    if low > high:
        raise ValueError(f"`low` must be <= `high`. Got low={low}, high={high}.")

    ad = adata.copy() if copy else adata

    X = ad.X

    # Store the original X into a new layer
    if sparse.issparse(X):
        ad.layers[layer_name] = X.copy()
        M = X.tocsr(copy=True)
        data = M.data

        mask_low = data < low
        mask_high = data > high
        mask_mid = (~mask_low) & (~mask_high)

        data[mask_low] = 0.0
        data[mask_high] = 1.0
        data[mask_mid] = float(mid)

        M.eliminate_zeros()
        ad.X = M
    else:
        ad.layers[layer_name] = np.array(X, copy=True).astype(dtype, copy=False)

        arr = np.array(X, copy=True)
        out = np.full(arr.shape, float(mid), dtype=dtype)
        out[arr < low] = 0.0
        out[arr > high] = 1.0

        ad.X = out

    return ad


def filter_adata_by_mutation_nnz(
    adata,
    min_nnz: int = 2,
    binarize_nonzero_to_one: bool = False,
    to_csr: bool = True,
    verbose: bool = True,
):
    """
    Filter AnnData columns (mutations) by the number of non-zero entries per column,
    and optionally binarize all non-zero values to 1.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object (cells x mutations) stored in adata.X.
    min_nnz : int, default=2
        Keep columns that have at least `min_nnz` non-zero entries.
    binarize_nonzero_to_one : bool, default=False
        If True, convert any non-zero value (e.g., 0.5 or 1) to 1.
        Note: for sparse matrices, this sets ALL stored data entries to 1.0,
        matching the original behavior (including the rare case of explicitly stored zeros).
    to_csr : bool, default=True
        If True and the resulting matrix is sparse, convert it to CSR format.
    verbose : bool, default=True
        If True, print basic stats and how many columns are kept.

    Returns
    -------
    adata_filt : AnnData
        A filtered copy of the input AnnData.
    """
    import numpy as np
    import scipy.sparse as sp

    X = adata.X
    n_cells, n_muts = adata.n_obs, adata.n_vars

    # 1) Compute per-column nnz on the original matrix (no full copy)
    if sp.issparse(X):
        # Fast paths for common formats
        if sp.isspmatrix_csr(X):
            # Count occurrences of each column index in CSR indices
            nnz_per_gene = np.bincount(X.indices, minlength=n_muts)
        elif sp.isspmatrix_csc(X):
            # In CSC, indptr encodes nnz per column
            nnz_per_gene = np.diff(X.indptr)
        else:
            nnz_per_gene = np.asarray(X.getnnz(axis=0)).ravel()

        nonzero_total = int(X.nnz)
    else:
        nnz_per_gene = np.count_nonzero(X, axis=0)
        nonzero_total = int(np.count_nonzero(X))

    keep_mask = nnz_per_gene >= int(min_nnz)
    n_keep = int(keep_mask.sum())

    if verbose:
        total_entries = n_cells * n_muts
        ratio = nonzero_total / total_entries if total_entries else float("nan")
        print(f"Input shape: {n_cells} cells x {n_muts} mutations")
        print(f"Non-zero ratio: {ratio:.6g}")
        print(f"Number of kept mutations (nnz>={min_nnz}): {n_keep}")

        # Same optional diagnostics as your original implementation
        if sp.issparse(X):
            vals = X.data
            count_1 = int(np.count_nonzero(np.isclose(vals, 1.0)))
            count_05 = int(np.count_nonzero(np.isclose(vals, 0.5)))
            print(f"Original sparse data counts: 1.0 -> {count_1}, 0.5 -> {count_05}")

    # 2) Slice AnnData once (avoid double slicing X)
    adata_filt = adata[:, keep_mask].copy()

    # 3) Optional binarization on the filtered submatrix only
    if binarize_nonzero_to_one:
        if sp.issparse(adata_filt.X):
            # Match original behavior: set ALL stored data entries to 1.0
            adata_filt.X.data[:] = 1.0
        else:
            adata_filt.X = (adata_filt.X != 0).astype(np.float32)

    # 4) Convert to CSR only if requested and needed
    if to_csr and sp.issparse(adata_filt.X) and (not sp.isspmatrix_csr(adata_filt.X)):
        adata_filt.X = adata_filt.X.tocsr()

    return adata_filt


def get_pcs(adata):
    """Return (PC matrix, cell name list).

    First try adata.obsm['X_pca'] (shape: n_cells × n_pcs).
    If it does not exist, try adata.uns['pca']['X'].
    If neither is available, raise a KeyError.
    """
    import numpy as np

    names = adata.obs_names.tolist()
    if hasattr(adata, 'obsm') and 'X_pca' in adata.obsm:
        X = adata.obsm['X_pca']
        return np.asarray(X), names
    if hasattr(adata, 'uns') and 'pca' in adata.uns and isinstance(adata.uns['pca'], dict) and 'X' in adata.uns['pca']:
        X = adata.uns['pca']['X']
        return np.asarray(X), names
    raise KeyError("PCA not found, please provide adata.obsm['X_pca'] or adata.uns['pca']['X']")


def plot_snv_pca_by_group(
    adata,
    pcx=1,
    pcy=2,
    size=6,
    alpha=0.7,
    cmap="viridis",
    group_by=None,
    group_colors=None,
    legend_loc="best",
    savepath=None,       
    dpi=300,              
    bbox_inches="tight",  
):
    """
    Scatter plot of two PCs colored by a specified obs column (or uncolored if group_by=None).

    Parameters
    ----------
    adata : AnnData
    pcx, pcy : int
        1-based indices (PC1=1).
    size : float
        Marker size.
    alpha : float
        Marker alpha.
    cmap : str
        Colormap name for continuous coloring.
    group_by : str | None
        Column name in adata.obs used for coloring. If None, plot points without coloring.
    group_colors : dict | list | tuple | None
        For categorical group_by:
        - None: use matplotlib default palette.
        - dict: mapping {category: color}.
        - list/tuple: colors aligned with category order.
        For continuous group_by, this is ignored (use cmap), unless you pass group_colors,
        in which case we force categorical coloring.
    legend_loc : str
        Legend location for categorical coloring.
    savepath : str | None
        If provided, save the figure to this path (supports .png/.pdf/.svg/.jpg etc).
    dpi : int
        DPI used when saving raster formats (e.g., png/jpg).
    bbox_inches : str
        bbox_inches argument passed to fig.savefig.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    pcs, names = get_pcs(adata)
    x = pcs[:, pcx - 1]
    y = pcs[:, pcy - 1]

    fig, ax = plt.subplots(figsize=(6, 5))

    # No coloring
    if group_by is None:
        ax.scatter(x, y, s=size, alpha=alpha)
        ax.set_xlabel(f"PC{pcx}")
        ax.set_ylabel(f"PC{pcy}")
        ax.set_title("PCA projection")
        fig.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, dpi=dpi, bbox_inches=bbox_inches)
        plt.show()
        plt.close(fig)
        return

    if group_by not in adata.obs:
        plt.close(fig)
        raise KeyError(f"`group_by='{group_by}'` not found in `adata.obs`.")

    g = adata.obs[group_by]

    is_cat = (
        pd.api.types.is_categorical_dtype(g)
        or pd.api.types.is_object_dtype(g)
        or pd.api.types.is_string_dtype(g)
    )
    is_continuous = (not is_cat) and pd.api.types.is_numeric_dtype(g) and (group_colors is None)

    if is_continuous:
        vals = np.asarray(g, dtype=float)
        sca = ax.scatter(x, y, c=vals, s=size, alpha=alpha, cmap=cmap)
        cbar = fig.colorbar(sca, ax=ax)
        cbar.set_label(group_by)
    else:
        # stable category order
        try:
            g_cat = g.astype("category")
            cats = list(g_cat.cat.categories)
            g_str = g_cat.astype(str).values
        except Exception:
            cats = sorted(set(map(str, g)))
            g_str = np.asarray(list(map(str, g)))

        # default palette (extend if needed)
        base = plt.rcParams.get("axes.prop_cycle", None)
        base_colors = base.by_key().get("color", []) if base is not None else []
        if len(base_colors) < len(cats):
            tab = plt.get_cmap("tab20")
            extra = [tab(i % tab.N) for i in range(len(cats))]
            palette = (base_colors + extra)[: len(cats)]
        else:
            palette = base_colors[: len(cats)]

        # build default color mapping
        color_map = {c: palette[i] for i, c in enumerate(cats)}

        # apply user-provided colors
        if group_colors is not None:
            if isinstance(group_colors, dict):
                # Allow keys to be int/float/str; match via string form first, then direct
                for c in cats:
                    if c in group_colors:
                        color_map[c] = group_colors[c]
                    else:
                        cs = str(c)
                        if cs in group_colors:
                            color_map[c] = group_colors[cs]
                        else:
                            # try numeric conversion match (e.g., "0" vs 0)
                            try:
                                ci = int(float(cs))
                                if ci in group_colors:
                                    color_map[c] = group_colors[ci]
                            except Exception:
                                pass
            else:
                if len(group_colors) != len(cats):
                    plt.close(fig)
                    raise ValueError(
                        f"`group_colors` length ({len(group_colors)}) must match "
                        f"number of categories ({len(cats)}): {cats}"
                    )
                color_map = {c: group_colors[i] for i, c in enumerate(cats)}

        # plot per category to create a legend
        for c in cats:
            mask = (g_str == str(c))
            ax.scatter(
                x[mask],
                y[mask],
                s=size,
                alpha=alpha,
                color=color_map[c],
                label=str(c),
            )

        ax.legend(title=group_by, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax.set_xlabel(f"PC{pcx}")
    ax.set_ylabel(f"PC{pcy}")
    ax.set_title(f"PCA projection colored by {group_by}")
    fig.subplots_adjust(right=0.78)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches=bbox_inches)

    plt.show()
    plt.close(fig)


def build_and_plot_nj(
    adata_sub,
    K_tree=10,
    K_jackknife=10,
    group_key="group",
    out_name="Non_SNV_Outgroup",
    out_group_label="Outgroup",
    metric="euclidean",
    show_leaf_labels=False,
    show_legend=True,
    group_colors=None,
    figsize=(10, 7),
    title_prefix="NJ tree",
    mark_outgroup_tip=True,
    outgroup_tip_size=90,
    y_pad_frac=0.08,
    y_pad_abs=0.0,
    savepath=None,
    print_tree_info=False,
    return_tree_info=False,
    leafcolor_onside=False,
    jackknife=False,
    jackknife_label_fmt="{:.0f}",
    jackknife_min_support=None,
):
    import os
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import distance
    from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
    from Bio import Phylo
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from io import StringIO

    def _x_positions(tree):
        depths = tree.depths()
        depths = depths if max(depths.values(), default=0) > 0 else tree.depths(unit_branch_lengths=True)
        return depths

    def _y_positions(tree):
        terms = tree.get_terminals()
        y = {t: i for i, t in enumerate(terms)}
        for cl in tree.get_nonterminals(order="postorder"):
            y[cl] = float(np.mean([y[ch] for ch in cl.clades]))
        return y

    def _newick_str(tree):
        buf = StringIO()
        Phylo.write(tree, buf, "newick")
        return buf.getvalue().strip()

    def _collect_tree_info(tree, out_name, unique_groups, group_key, K_tree, K_jackknife, metric, jk_info=None):
        terminals = tree.get_terminals()
        nonterminals = tree.get_nonterminals()
        n_tips = len(terminals)
        n_internal = len(nonterminals)

        bl = []
        for cl in tree.find_clades(order="preorder"):
            if cl.branch_length is not None:
                bl.append(float(cl.branch_length))
        bl = np.asarray(bl, dtype=float) if len(bl) else np.asarray([], dtype=float)

        depths = tree.depths()
        if max(depths.values(), default=0) == 0:
            depths = tree.depths(unit_branch_lengths=True)

        tip_depths = {t.name: float(depths.get(t, 0.0)) for t in terminals if t.name is not None}
        outgroup_depth = tip_depths.get(out_name, None)

        info = {
            "meta": {
                "K_tree": int(K_tree),
                "K_jackknife": int(K_jackknife) if K_jackknife is not None else None,
                "metric": str(metric),
                "group_key": str(group_key),
                "groups": list(unique_groups),
                "outgroup_name": str(out_name),
                "jackknife": bool(jackknife),
            },
            "counts": {
                "n_tips": int(n_tips),
                "n_internal": int(n_internal),
                "n_clades_total": int(n_tips + n_internal),
            },
            "branch_length": {
                "n_with_length": int(bl.size),
                "min": float(bl.min()) if bl.size else None,
                "max": float(bl.max()) if bl.size else None,
                "mean": float(bl.mean()) if bl.size else None,
                "median": float(np.median(bl)) if bl.size else None,
                "sum": float(bl.sum()) if bl.size else None,
            },
            "tip_depths": tip_depths,
            "outgroup_depth": outgroup_depth,
            "newick": _newick_str(tree),
        }
        if jk_info is not None:
            info["jackknife_info"] = jk_info
        return info

    def _build_nj_tree_from_cols(pcs_aug, names_aug, out_name, cols, metric):
        pcs_use = np.asarray(pcs_aug[:, cols], dtype=float)
        D = distance.squareform(distance.pdist(pcs_use, metric=metric))
        tri = [D[i, : i + 1].tolist() for i in range(len(names_aug))]
        dm = DistanceMatrix(names=names_aug, matrix=tri)
        t = DistanceTreeConstructor().nj(dm)

        try:
            out_clade = next(t.find_clades(name=out_name))
        except StopIteration:
            raise ValueError(f"Outgroup '{out_name}' not found in tree tips.")
        t.root_with_outgroup(out_clade)
        t.ladderize()
        return t

    def _canonical_split(clade_obj, all_taxa, out_name):
        leaves = {t.name for t in clade_obj.get_terminals() if t.name is not None and t.name != out_name}
        if len(leaves) == 0 or len(leaves) == len(all_taxa):
            return None
        if len(leaves) > (len(all_taxa) / 2.0):
            leaves = set(all_taxa) - leaves
        return frozenset(leaves)

    def _splits_from_tree(tree_obj, all_taxa, out_name):
        splits = set()
        for cl in tree_obj.get_nonterminals():
            if cl is tree_obj.root:
                continue
            s = _canonical_split(cl, all_taxa, out_name)
            if s is not None:
                splits.add(s)
        return splits

    def _add_jackknife_support(tree_obj, pcs_aug, names_aug, out_name, cols_jk, metric):
        if len(cols_jk) <= 1:
            warnings.warn("Too few PCA dimensions available for jackknife.")
            return {"n_reps": 0, "mode": "leave_one_out", "cols_used": list(cols_jk)}

        all_taxa = set([n for n in names_aug if n != out_name])

        clade_to_split = {}
        for cl in tree_obj.get_nonterminals():
            if cl is tree_obj.root:
                continue
            s = _canonical_split(cl, all_taxa, out_name)
            if s is not None:
                clade_to_split[cl] = s

        if not clade_to_split:
            warnings.warn("No informative internal splits found for jackknife annotation.")
            return {"n_reps": 0, "mode": "leave_one_out", "cols_used": list(cols_jk)}

        split_counts = {s: 0 for s in set(clade_to_split.values())}

        rep_cols = []
        for drop_c in cols_jk:
            cols = [c for c in cols_jk if c != drop_c]
            if len(cols) >= 1:
                rep_cols.append(cols)

        n_reps = len(rep_cols)
        if n_reps == 0:
            warnings.warn("Jackknife produced zero replicates.")
            return {"n_reps": 0, "mode": "leave_one_out", "cols_used": list(cols_jk)}

        for cols in rep_cols:
            t_jk = _build_nj_tree_from_cols(pcs_aug, names_aug, out_name, cols, metric)
            splits_jk = _splits_from_tree(t_jk, all_taxa, out_name)
            for s in split_counts:
                if s in splits_jk:
                    split_counts[s] += 1

        for cl, s in clade_to_split.items():
            cl.confidence = 100.0 * split_counts[s] / float(n_reps)

        return {"n_reps": int(n_reps), "mode": "leave_one_out", "cols_used": list(cols_jk)}

    if "X_pca" not in adata_sub.obsm:
        raise KeyError("adata_sub.obsm['X_pca'] not found.")
    if "PCs" not in adata_sub.varm:
        raise KeyError("adata_sub.varm['PCs'] not found.")
    if group_key not in adata_sub.obs:
        raise KeyError(f"'{group_key}' not found in adata_sub.obs.")

    names_sub = adata_sub.obs_names.tolist()
    groups_sub = adata_sub.obs[group_key].astype(str).to_numpy()

    pcs_sub = np.asarray(adata_sub.obsm["X_pca"])
    X_sub = adata_sub.X

    mu = np.asarray(X_sub.mean(axis=0)).ravel()
    W = np.asarray(adata_sub.varm["PCs"])
    out_pcs = (-mu @ W).reshape(1, -1)

    pcs_aug = np.vstack([pcs_sub, out_pcs])
    names_aug = names_sub + [out_name]

    K_tree_eff = int(min(int(K_tree), pcs_aug.shape[1]))
    if K_tree_eff < 1:
        raise ValueError("K_tree is too small or PCA dimensionality is insufficient.")

    cols_ref = list(range(K_tree_eff))
    tree = _build_nj_tree_from_cols(pcs_aug, names_aug, out_name, cols_ref, metric)

    jk_info = None
    if jackknife:
        if K_jackknife is None:
            K_jk_eff = K_tree_eff
        else:
            K_jk_eff = int(min(int(K_jackknife), pcs_aug.shape[1]))
        cols_jk = list(range(max(K_jk_eff, 0)))
        jk_info = _add_jackknife_support(tree, pcs_aug, names_aug, out_name, cols_jk, metric)

    unique_groups = list(dict.fromkeys(groups_sub.tolist()))

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    default_cycle = default_cycle if len(default_cycle) else ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

    if group_colors is None:
        color_list = default_cycle
    elif isinstance(group_colors, (list, tuple)):
        color_list = list(group_colors)
    else:
        color_list = default_cycle

    group_to_color = {g: color_list[i % len(color_list)] for i, g in enumerate(unique_groups)}
    group_to_color[out_group_label] = "black"

    if isinstance(group_colors, dict):
        group_to_color.update(group_colors)
        group_to_color[out_group_label] = "black"

    name_to_group = dict(zip(names_sub, groups_sub))
    name_to_group[out_name] = out_group_label

    label_colors = {nm: group_to_color[str(name_to_group[nm])] for nm in names_aug}

    for cl in tree.get_terminals():
        cl.color = label_colors.get(cl.name, "black")

    def label_func(clade):
        return clade.name if (show_leaf_labels and clade.is_terminal()) else None

    def branch_label_func(clade):
        if (not jackknife) or clade.is_terminal():
            return None
        conf = getattr(clade, "confidence", None)
        if conf is None:
            return None
        if (jackknife_min_support is not None) and (conf < float(jackknife_min_support)):
            return None
        try:
            return jackknife_label_fmt.format(conf)
        except Exception:
            return f"{conf:.0f}"

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    title = f"{title_prefix} | K_tree={K_tree_eff} | colored by {group_key} | rooted by {out_name}"
    if jackknife:
        K_jk_disp = K_tree_eff if (K_jackknife is None) else int(min(int(K_jackknife), pcs_aug.shape[1]))
        if jk_info is not None and jk_info.get("n_reps", 0) > 0:
            title = title + f" | jackknife=LOO(K_jackknife={K_jk_disp}, n={jk_info['n_reps']})"
        else:
            title = title + f" | jackknife=LOO(K_jackknife={K_jk_disp})"
    ax.set_title(title)

    Phylo.draw(
        tree,
        axes=ax,
        label_func=label_func,
        label_colors=label_colors,
        branch_labels=branch_label_func if jackknife else None,
        do_show=False,
    )

    y0, y1 = ax.get_ylim()
    yr = (y1 - y0)
    pad = yr * float(y_pad_frac) + float(y_pad_abs)
    ax.set_ylim(y0 - pad, y1 + pad)

    if mark_outgroup_tip:
        x_pos = _x_positions(tree)
        y_pos = _y_positions(tree)
        out_term = next(tree.find_clades(name=out_name))
        ax.scatter(
            [x_pos[out_term]],
            [y_pos[out_term]],
            s=outgroup_tip_size,
            facecolors="white",
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
            clip_on=False,
        )

    if show_legend:
        handles = [Patch(facecolor=group_to_color[g], edgecolor="none", label=g) for g in unique_groups]
        out_handle = Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=8,
            label=out_group_label,
        )
        handles.append(out_handle)
        ax.legend(handles=handles, title=group_key, loc="best", frameon=False)

    if leafcolor_onside:
        label_y = {}
        for txt in ax.texts:
            name = txt.get_text()
            if name in name_to_group:
                label_y[name] = txt.get_position()[1]

        terminals = tree.get_terminals()
        y_pos_fallback = _y_positions(tree)

        ys = []
        bar_colors = []
        for t in terminals:
            name = t.name
            y = float(label_y[name]) if name in label_y else float(y_pos_fallback[t])
            ys.append(y)
            bar_colors.append(label_colors.get(name, "black"))

        ys = np.asarray(ys, dtype=float)

        order = np.argsort(ys)
        ys = ys[order]
        bar_colors = [bar_colors[i] for i in order]

        if len(ys) > 1:
            if len(ys) > 2:
                dy0 = float(np.diff(ys[1:]).mean())
            else:
                dy0 = float(ys[1] - ys[0])
            ys_aligned = ys.copy()
            base = ys[1]
            for i in range(len(ys_aligned)):
                ys_aligned[i] = base + i * dy0
        else:
            ys_aligned = ys

        bar_h = float(np.diff(ys_aligned).min()) if len(ys_aligned) > 1 else 1.0

        x0, x1 = ax.get_xlim()
        span = x1 - x0 if x1 > x0 else 1.0
        gap = 0.01 * span
        width = 0.02 * span
        left = x1 + gap

        for y, c in zip(ys_aligned, bar_colors):
            ax.barh(y, width, height=bar_h, left=left, color=c, edgecolor="none", align="center")

        ax.set_xlim(x0, left + width + gap)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", left=False, right=False)

    plt.tight_layout()

    if savepath is not None:
        try:
            savepath = os.fspath(savepath)
            parent = os.path.dirname(savepath)
            if parent:
                os.makedirs(parent, exist_ok=True)
            fig.savefig(savepath, dpi=300, bbox_inches="tight")
        except Exception as e:
            warnings.warn(f"Failed to save figure to {savepath!r}: {e}")

    plt.show()

    tree_info = None
    if print_tree_info or return_tree_info:
        tree_info = _collect_tree_info(
            tree,
            out_name,
            unique_groups,
            group_key,
            K_tree_eff,
            (K_tree_eff if K_jackknife is None else int(min(int(K_jackknife), pcs_aug.shape[1]))),
            metric,
            jk_info=jk_info,
        )

    if print_tree_info:
        meta = tree_info["meta"]
        counts = tree_info["counts"]
        bl = tree_info["branch_length"]

        print("=== NJ Tree Info ===")
        print(f"K_tree={meta['K_tree']}, K_jackknife={meta['K_jackknife']}, metric={meta['metric']}, group_key={meta['group_key']}")
        print(f"groups={meta['groups']}")
        print(f"outgroup={meta['outgroup_name']}, outgroup_depth={tree_info['outgroup_depth']}")
        print(f"n_tips={counts['n_tips']}, n_internal={counts['n_internal']}, n_total_clades={counts['n_clades_total']}")
        print(
            "branch_length:",
            f"n={bl['n_with_length']}, min={bl['min']}, max={bl['max']}, mean={bl['mean']}, median={bl['median']}, sum={bl['sum']}",
        )
        if jackknife and jk_info is not None:
            print(f"jackknife: n_reps={jk_info.get('n_reps', 0)}, mode={jk_info.get('mode', '')}, cols_used={jk_info.get('cols_used', [])}")
        print("newick:")
        print(tree_info["newick"])

    if return_tree_info:
        return tree, tree_info
    return tree
