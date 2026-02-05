def plot_cmb_cent(
    adata,
    coor,
    cmb_info,
    size_by_ncells=True,
    cell_alpha=0.3,
    cell_size=30,
    title="CMB distribution on spatial coordinates",
    group_colors=None,
    show_group_legend=True,
    figsize=(8, 6),
    aspect="auto",         
    savepath=None,         
    show=True,              
    dpi=300,                
    transparent=False,      
    bbox_inches="tight",    
    show_labels=True,       
):

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if isinstance(coor, np.ndarray):
        coor = pd.DataFrame(coor, columns=["x", "y"])
    elif isinstance(coor, pd.DataFrame):
        coor = coor.rename(columns={coor.columns[0]: "x", coor.columns[1]: "y"})

    coor = coor.copy()
    coor["group"] = cmb_info

    def _resolve_group_colors(unique_groups, user_colors):
        
        default_tab20 = plt.cm.tab20(np.linspace(0, 1, len(unique_groups)))
        default_map = dict(zip(unique_groups, default_tab20))

        if user_colors is None:
            return default_map

        if isinstance(user_colors, dict):
            normalized = {str(k): v for k, v in user_colors.items()}
            return {g: normalized.get(g, default_map[g]) for g in unique_groups}

        if isinstance(user_colors, (list, tuple, np.ndarray)):
            if len(user_colors) == 0:
                return default_map
            seq = list(user_colors)
            colors = [seq[i % len(seq)] for i in range(len(unique_groups))]
            return dict(zip(unique_groups, colors))

        if isinstance(user_colors, str):
            try:
                cmap = plt.cm.get_cmap(user_colors)
                colors = cmap(np.linspace(0, 1, len(unique_groups)))
                return dict(zip(unique_groups, colors))
            except Exception:
                return default_map

        return default_map

    fig, ax = plt.subplots(figsize=figsize)
    if show_group_legend:
        fig.subplots_adjust(right=0.75)

    unique_groups = coor["group"].astype(str).unique().tolist()
    unique_groups = sorted(unique_groups)

    cluster_color_map = _resolve_group_colors(unique_groups, group_colors)

    for cluster in unique_groups:
        subset = coor[coor["group"].astype(str) == cluster]
        ax.scatter(
            subset["x"],
            subset["y"],
            c=[cluster_color_map[cluster]],
            label=cluster,
            s=cell_size,
            alpha=cell_alpha,
        )

    obs = adata.obs
    required_cols = ["x_center", "y_center", "n_cells"]
    missing = [c for c in required_cols if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing columns in adata.obs: {missing}")

    mask = obs[required_cols].notna().all(axis=1)
    d = obs.loc[mask, required_cols].astype(float)

    x = d["x_center"].to_numpy()
    y = d["y_center"].to_numpy()
    labels = adata.obs_names[mask].astype(str).to_numpy()

    if size_by_ncells:
        sizes = d["n_cells"].clip(lower=1).to_numpy()
        denom = np.sqrt(sizes.max()) if sizes.size and sizes.max() > 0 else 1.0
        s = 20 + 180 * (np.sqrt(sizes) / denom)
    else:
        s = np.full(len(d), 120.0, dtype=float)

    center_colors = [
        cluster_color_map.get(str(lb), (0.2, 0.2, 0.2, 1.0)) for lb in labels
    ]

    ax.scatter(
        x,
        y,
        s=s,
        c=center_colors,
        alpha=0.95,
        marker="o",
        edgecolors="k",
        linewidths=0.6,
    )

    if show_labels:
        try:
            from adjustText import adjust_text

            texts = [
                ax.text(xx, yy, str(lb), fontsize=8, color="black")
                for xx, yy, lb in zip(x, y, labels)
            ]
            if texts:
                adjust_text(
                    texts,
                    x=x,
                    y=y,
                    expand_points=(1.2, 1.4),
                    expand_text=(1.2, 1.4),
                    only_move={"points": "y", "text": "xy"},
                    arrowprops=dict(
                        arrowstyle="-", lw=0.8, alpha=0.9, color="black"
                    ),
                )
        except ImportError:
            for xx, yy, lb in zip(x, y, labels):
                ax.annotate(
                    str(lb),
                    xy=(xx, yy),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    arrowprops=dict(
                        arrowstyle="-", lw=0.8, alpha=0.9, color="black"
                    ),
                )

    if show_group_legend:
        handles = [
            mpatches.Patch(color=cluster_color_map[cluster], label=cluster)
            for cluster in unique_groups
        ]
        ax.legend(handles=handles, title="Group", bbox_to_anchor=(1, 1), loc="upper left")

    ax.set_title(title)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")

    try:
        if isinstance(aspect, (int, float)):
            ax.set_aspect(float(aspect))
        elif isinstance(aspect, str):
            ax.set_aspect(aspect)
    except Exception:
        pass

    plt.tight_layout()

    if savepath:
        try:
            parent = os.path.dirname(savepath)
            if parent:
                os.makedirs(parent, exist_ok=True)
        except Exception:
            pass

        fig.savefig(
            savepath,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
            metadata={"Title": title},
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_obs_with_cmb_cent(
    adata,
    coor,
    cmb_info,
    group_by,
    cmap="viridis",
    ntop=5,
    size_by_ncells=True,
    cell_alpha=0.3,
    cell_size=30,
    savepath=None,
    title=None,
    group_colors=None,
    center_colors=None,
    tohighl=None,
    dpi=300,
    show_group_legend=True,
    show_center_legend=True,
    figsize=(9, 6),
    bbox_inches="tight",
    transparent=False,
    show=True,
    aspect="auto",  
    show_labels=True,
):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os

    if isinstance(coor, np.ndarray):
        coor = pd.DataFrame(coor, columns=["x", "y"])
    elif isinstance(coor, pd.DataFrame):
        coor = coor.rename(columns={coor.columns[0]: "x", coor.columns[1]: "y"})
    else:
        raise TypeError("`coor` must be a np.ndarray or pd.DataFrame")

    coor["group"] = cmb_info

    def _resolve_group_colors(unique_groups, user_colors, fallback_cmap="tab20"):
        """Build {group -> color} from None/dict/list/colormap-name."""
        unique_groups = [str(g) for g in unique_groups]
        default_cols = plt.cm.get_cmap(fallback_cmap)(np.linspace(0, 1, len(unique_groups)))
        default_map = dict(zip(unique_groups, default_cols))

        if user_colors is None:
            return default_map

        if isinstance(user_colors, dict):
            normalized = {str(k): v for k, v in user_colors.items()}
            return {g: normalized.get(g, default_map[g]) for g in unique_groups}

        if isinstance(user_colors, (list, tuple, np.ndarray)):
            seq = list(user_colors)
            if len(seq) == 0:
                return default_map
            colors = [seq[i % len(seq)] for i in range(len(unique_groups))]
            return dict(zip(unique_groups, colors))

        if isinstance(user_colors, str):
            try:
                cm = plt.cm.get_cmap(user_colors)
                colors = cm(np.linspace(0, 1, len(unique_groups)))
                return dict(zip(unique_groups, colors))
            except Exception:
                return default_map

        return default_map

    def _infer_numeric(series: pd.Series):
        """Return (is_numeric, numeric_series_or_none)."""
        s = series
        
        if pd.api.types.is_numeric_dtype(s):
            return True, pd.to_numeric(s, errors="coerce")

        coerced = pd.to_numeric(s, errors="coerce")
        non_na = s.notna()
        if non_na.sum() == 0:
            return False, None

        ratio = coerced[non_na].notna().mean()
        if ratio >= 0.90:
            return True, coerced
        return False, None

    unique_groups = coor["group"].astype(str).unique().tolist()
    unique_groups = sorted(unique_groups)
    cluster_color_map = _resolve_group_colors(unique_groups, group_colors, fallback_cmap="tab20")

    if tohighl is not None:
        if isinstance(tohighl, (list, tuple, set, np.ndarray, pd.Series)):
            highlight_groups = {str(g) for g in tohighl}
        else:
            highlight_groups = {str(tohighl)}
        cluster_color_map = {
            g: cluster_color_map[g] if g in highlight_groups else "#999999"
            for g in unique_groups
        }

    obs = adata.obs
    if group_by not in obs.columns:
        raise KeyError(f"Cannot find column in adata.obs: {group_by}")

    required_cols = ["x_center", "y_center", "n_cells"]
    missing = [c for c in required_cols if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing columns in adata.obs: {missing}")

    mask = obs[required_cols].notna().all(axis=1)

    gb_raw = obs.loc[mask, group_by]
    is_num, gb_num = _infer_numeric(gb_raw)

    if is_num:

        gb_num = gb_num.astype(float)
        keep = gb_num.notna()
        gb_vals = gb_num[keep].to_numpy()
        mask_idx = gb_num.index[keep]
        
    else:

        gb_cat = gb_raw.astype("object")
        keep = gb_cat.notna()
        gb_vals = gb_cat[keep].astype(str).to_numpy()
        mask_idx = gb_cat.index[keep]

    d = obs.loc[mask_idx, required_cols].astype(float)
    x = d["x_center"].to_numpy()
    y = d["y_center"].to_numpy()
    labels = obs.index.astype(str)
    labels = labels[obs.index.get_indexer(mask_idx)].to_numpy()

    if size_by_ncells and len(d):
        sizes = d["n_cells"].clip(lower=1).to_numpy()
        denom = np.sqrt(sizes.max()) if sizes.size and sizes.max() > 0 else 1.0
        s = 20 + 180 * (np.sqrt(sizes) / denom)
    else:
        s = np.full(len(d), 120.0, dtype=float)

    top_idx = []
    if is_num and ntop and len(gb_vals):
        valid = ~np.isnan(gb_vals)
        if valid.any():
            idx_valid = np.where(valid)[0]
            top_rel = np.argsort(gb_vals[idx_valid])[-int(ntop):]
            top_idx = idx_valid[top_rel].tolist()

    if title is None:
        title = f"Mapping {group_by} on spatial coordinates"

    fig, ax = plt.subplots(figsize=figsize)
    if show_group_legend:
        fig.subplots_adjust(right=0.75)

    for cluster in unique_groups:
        subset = coor[coor["group"].astype(str) == cluster]
        ax.scatter(
            subset["x"],
            subset["y"],
            c=[cluster_color_map[cluster]],
            label=cluster,
            s=cell_size,
            alpha=cell_alpha,
        )

    if len(d):
        if is_num:
            sc = ax.scatter(
                x,
                y,
                s=s,
                c=gb_vals,
                cmap=cmap,
                alpha=0.85,
                marker="o",
                edgecolors="k",
                linewidths=0.6,
            )
            cb = fig.colorbar(sc, ax=ax, pad=0.1, location="left")
            cb.set_label(str(group_by))

            if show_labels:
                try:
                    from adjustText import adjust_text
                    texts_top, texts_oth = [], []
                    for i, (xx, yy, lb, v) in enumerate(zip(x, y, labels, gb_vals)):
                        is_top_and_pos = (i in top_idx) and (not np.isnan(v)) and (v > 0)
                        if is_top_and_pos:
                            texts_top.append(ax.text(xx, yy, str(lb), fontsize=8, color="black"))
                        else:
                            texts_oth.append(ax.text(xx, yy, str(lb), fontsize=8, color="gray"))

                    if texts_top:
                        adjust_text(
                            texts_top,
                            x=x,
                            y=y,
                            expand_points=(1.2, 1.4),
                            expand_text=(1.2, 1.4),
                            only_move={"points": "y", "text": "xy"},
                            arrowprops=dict(arrowstyle="-", lw=0.8, alpha=0.9, color="black"),
                        )
                    if texts_oth:
                        adjust_text(
                            texts_oth,
                            x=x,
                            y=y,
                            expand_points=(1.1, 1.2),
                            expand_text=(1.1, 1.2),
                            only_move={"points": "y", "text": "xy"},
                            arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.6, color="gray"),
                        )
                except ImportError:
                    for i, (xx, yy, lb, v) in enumerate(zip(x, y, labels, gb_vals)):
                        is_top_and_pos = (i in top_idx) and (not np.isnan(v)) and (v > 0)
                        color = "black" if is_top_and_pos else "gray"
                        ax.annotate(
                            str(lb),
                            xy=(xx, yy),
                            xytext=(2, 2),
                            textcoords="offset points",
                            fontsize=8,
                            color=color,
                            arrowprops=dict(
                                arrowstyle="-",
                                lw=0.6 if color == "gray" else 0.8,
                                alpha=0.6 if color == "gray" else 0.9,
                                color=color,
                            ),
                        )

        else:
            
            center_cats = pd.unique(pd.Series(gb_vals))
            center_cats = [str(c) for c in center_cats.tolist()]
            center_cats = sorted(center_cats)

            center_color_map = _resolve_group_colors(
                center_cats,
                center_colors,
                fallback_cmap=("tab20" if center_colors is None else "tab20"),
            )

            center_point_colors = [center_color_map[str(v)] for v in gb_vals]

            ax.scatter(
                x,
                y,
                s=s,
                c=center_point_colors,
                alpha=0.9,
                marker="o",
                edgecolors="k",
                linewidths=0.6,
            )

            if show_labels:
                try:
                    from adjustText import adjust_text
                    texts = [ax.text(xx, yy, str(lb), fontsize=8, color="gray")
                             for xx, yy, lb in zip(x, y, labels)]
                    if texts:
                        adjust_text(
                            texts,
                            x=x,
                            y=y,
                            expand_points=(1.1, 1.2),
                            expand_text=(1.1, 1.2),
                            only_move={"points": "y", "text": "xy"},
                            arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.6, color="gray"),
                        )
                except ImportError:
                    for xx, yy, lb in zip(x, y, labels):
                        ax.annotate(
                            str(lb),
                            xy=(xx, yy),
                            xytext=(2, 2),
                            textcoords="offset points",
                            fontsize=8,
                            color="gray",
                            arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.6, color="gray"),
                        )

            if show_center_legend:
                center_handles = [
                    mpatches.Patch(color=center_color_map[c], label=c) for c in center_cats
                ]
                leg2 = ax.legend(
                    handles=center_handles,
                    title=str(group_by),
                    loc="upper left",
                    bbox_to_anchor=(0.0, 1.0),
                    frameon=True,
                )
                ax.add_artist(leg2)

    else:
        ax.text(
            0.5,
            0.5,
            "No centers available (missing x_center/y_center/n_cells or group_by is NA).",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    if show_group_legend:
        handles = [
            mpatches.Patch(color=cluster_color_map[cluster], label=cluster)
            for cluster in unique_groups
        ]
        ax.legend(handles=handles, title="Group", bbox_to_anchor=(1, 1), loc="upper left")

    ax.set_title(title)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")

    try:
        if isinstance(aspect, (int, float)):
            ax.set_aspect(float(aspect))
        elif isinstance(aspect, str):
            ax.set_aspect(aspect)
    except Exception:
        pass

    if savepath:
        try:
            parent = os.path.dirname(savepath)
            if parent:
                os.makedirs(parent, exist_ok=True)
        except Exception:
            pass
        fig.savefig(
            savepath,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
            metadata={"Title": title},
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_snv_with_cmb_cent(
    adata,
    coor,
    cmb_info,
    snv,
    cmap='Reds',
    ntop=5,
    size_by_ncells=True,
    cell_alpha=0.3,
    cell_size=30,
    savepath=None,
    title="Mapping SNV on spatial coordinates",
    group_colors=None,
    tohighl=None,
    min_depth=20,
    depth_layer='depth',
    dpi=300,
    show_group_legend=True,
    figsize=(9, 6),
    bbox_inches='tight',
    transparent=False,
    show=True,
    aspect='auto',  
    show_labels=True,
):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import scipy.sparse as sp
    import os

    if isinstance(coor, np.ndarray):
        coor = pd.DataFrame(coor, columns=["x", "y"])
    elif isinstance(coor, pd.DataFrame):
        coor = coor.rename(columns={coor.columns[0]: "x", coor.columns[1]: "y"})

    coor["group"] = cmb_info

    def _resolve_group_colors(unique_groups, user_colors):
        """Build a color map {group -> color} from different user input types.
        Fallback to tab20 for anything missing/invalid.
        """
        default_tab20 = plt.cm.tab20(np.linspace(0, 1, len(unique_groups)))
        default_map = dict(zip(unique_groups, default_tab20))

        if user_colors is None:
            return default_map

        if isinstance(user_colors, dict):
            normalized = {str(k): v for k, v in user_colors.items()}
            return {g: normalized.get(g, default_map[g]) for g in unique_groups}

        if isinstance(user_colors, (list, tuple, np.ndarray)):
            if len(user_colors) == 0:
                return default_map
            seq = list(user_colors)
            colors = [seq[i % len(seq)] for i in range(len(unique_groups))]
            return dict(zip(unique_groups, colors))

        if isinstance(user_colors, str):
            try:
                cmap_local = plt.cm.get_cmap(user_colors)
                colors = cmap_local(np.linspace(0, 1, len(unique_groups)))
                return dict(zip(unique_groups, colors))
            except Exception:
                return default_map

        return default_map

    unique_groups = coor["group"].astype(str).unique().tolist()
    unique_groups = sorted(unique_groups)
    cluster_color_map = _resolve_group_colors(unique_groups, group_colors)

    if tohighl is not None:

        if isinstance(tohighl, (list, tuple, set, np.ndarray, pd.Series)):
            highlight_groups = {str(g) for g in tohighl}
        else:
            highlight_groups = {str(tohighl)}

        cluster_color_map = {
            g: cluster_color_map[g] if g in highlight_groups else "#999999"
            for g in unique_groups
        }

    if snv not in adata.var_names:
        raise KeyError(f"Cannot find feature in var_names: {snv}")
    col = adata.var_names.get_loc(snv)
    X = adata.X
    vals = X[:, col].toarray().ravel() if sp.issparse(X) else np.asarray(X[:, col]).ravel()

    obs = adata.obs
    required_cols = ['x_center', 'y_center', 'n_cells']
    missing = [c for c in required_cols if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing columns in adata.obs: {missing}")

    if depth_layer not in adata.layers:
        raise KeyError(f"Missing layer in adata.layers: '{depth_layer}'")
    depth_mat = adata.layers[depth_layer]
    depth_vals_col = (
        depth_mat[:, col].toarray().ravel() if sp.issparse(depth_mat)
        else np.asarray(depth_mat[:, col]).ravel()
    )

    mask = obs[required_cols].notna().all(axis=1)
    if min_depth is not None:
        mask = mask & (depth_vals_col >= float(min_depth))

    d = obs.loc[mask, required_cols].astype(float)
    vals_plot = np.asarray(vals)[mask]
    x = d['x_center'].to_numpy()
    y = d['y_center'].to_numpy()
    labels = adata.obs_names[mask].astype(str).to_numpy()

    if size_by_ncells and len(d):
        sizes = d['n_cells'].clip(lower=1).to_numpy()
        denom = np.sqrt(sizes.max()) if sizes.size and sizes.max() > 0 else 1.0
        s = 20 + 180 * (np.sqrt(sizes) / denom)
    else:
        s = np.full(len(d), 120.0, dtype=float)

    valid = ~np.isnan(vals_plot)
    top_idx = []
    if ntop > 0 and valid.any():
        idx_valid = np.where(valid)[0]
        top_rel = np.argsort(vals_plot[idx_valid])[-ntop:]
        top_idx = idx_valid[top_rel]

    fig, ax = plt.subplots(figsize=figsize)
    if show_group_legend:
        fig.subplots_adjust(right=0.75)

    for cluster in unique_groups:
        subset = coor[coor["group"].astype(str) == cluster]
        ax.scatter(
            subset["x"],
            subset["y"],
            c=[cluster_color_map[cluster]],
            label=cluster,
            s=cell_size,
            alpha=cell_alpha,
        )

    if len(d):
        sc = ax.scatter(
            x,
            y,
            s=s,
            c=vals_plot,
            cmap=cmap,
            alpha=0.85,
            marker='o',
            edgecolors='k',
            linewidths=0.6,
        )

        cb = fig.colorbar(sc, ax=ax, pad=0.1, location="left")
        cb.set_label(f'{snv} ratio')

        if show_labels:
            try:
                from adjustText import adjust_text
                texts_top, texts_oth = [], []
                for i, (xx, yy, lb, v) in enumerate(zip(x, y, labels, vals_plot)):
                    # only top_idx and ratio > 0 are black
                    is_top_and_nonzero = (i in top_idx) and (not np.isnan(v)) and (v > 0)
                    if is_top_and_nonzero:
                        texts_top.append(ax.text(xx, yy, str(lb), fontsize=8, color='black'))
                    else:
                        texts_oth.append(ax.text(xx, yy, str(lb), fontsize=8, color='gray'))

                if texts_top:
                    adjust_text(
                        texts_top,
                        x=x,
                        y=y,
                        expand_points=(1.2, 1.4),
                        expand_text=(1.2, 1.4),
                        only_move={'points': 'y', 'text': 'xy'},
                        arrowprops=dict(arrowstyle='-', lw=0.8, alpha=0.9, color='black'),
                    )
                if texts_oth:
                    adjust_text(
                        texts_oth,
                        x=x,
                        y=y,
                        expand_points=(1.1, 1.2),
                        expand_text=(1.1, 1.2),
                        only_move={'points': 'y', 'text': 'xy'},
                        arrowprops=dict(arrowstyle='-', lw=0.6, alpha=0.6, color='gray'),
                    )
            except ImportError:
                for i, (xx, yy, lb, v) in enumerate(zip(x, y, labels, vals_plot)):
                    is_top_and_nonzero = (i in top_idx) and (not np.isnan(v)) and (v > 0)
                    color = 'black' if is_top_and_nonzero else 'gray'
                    ax.annotate(
                        str(lb),
                        xy=(xx, yy),
                        xytext=(2, 2),
                        textcoords='offset points',
                        fontsize=8,
                        color=color,
                        arrowprops=dict(
                            arrowstyle='-',
                            lw=0.6 if color == 'gray' else 0.8,
                            alpha=0.6 if color == 'gray' else 0.9,
                            color=color,
                        ),
                    )
    else:
        ax.text(
            0.5,
            0.5,
            'No centers pass the depth filter.',
            ha='center',
            va='center',
            transform=ax.transAxes,
        )

    if show_group_legend:
        handles = [
            mpatches.Patch(color=cluster_color_map[cluster], label=cluster)
            for cluster in unique_groups
        ]
        ax.legend(handles=handles, title="Group", bbox_to_anchor=(1, 1), loc='upper left')

    ax.set_title(title)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")

    try:
        if isinstance(aspect, (int, float)):
            ax.set_aspect(float(aspect))
        elif isinstance(aspect, str):
            ax.set_aspect(aspect)
    except Exception:
        pass

    if savepath:
        try:
            parent = os.path.dirname(savepath)
            if parent:
                os.makedirs(parent, exist_ok=True)
        except Exception:
            pass

        metadata = {"Title": title}
        
        fig.savefig(
            savepath,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
            metadata=metadata,
        )

    if show:
        import matplotlib.pyplot as plt  
        plt.show()
    else:
        import matplotlib.pyplot as plt 
        plt.close(fig)

    return fig, ax

def plot_cmb_cent_vector(
    adata,
    coor,
    cell_info,
    group_colors=None,
    cell_group_col="group",
    coor_cell_col=None,
    show_cells=True,
    show_vector=True,
    root_cells=None,
    end_cells=None,
    cell_alpha=0.3,
    cell_size=20,
    size_by_ncells=True,
    show_labels=True,
    show_group_legend=False,
    figsize=(9,6),
    aspect="auto",
    vector_span_frac=0.12,
    title="CMB centers + vector field",
    savepath=None,
    dpi=300,
    transparent=False,
    bbox_inches="tight",
    show=True,
    plot_conn_graph_by=None,      # None, "Heatmap", or "Network"
    conn_graph_figsize=(5,5),     # Figure size for the connectivity graph
    heatmap_sort_labels=True,     # Whether to sort labels alphabetically in heatmap
    network_edge_alpha=0.96,       # Base alpha for network edges
    ##network_show_labels=False,    # Whether to show labels in network graph
):
   
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    import scipy.sparse as sp
    from scipy.sparse. csgraph import dijkstra

    if (root_cells is not None) and (end_cells is not None):
        raise ValueError("root_cells and end_cells cannot both be not None.")
    if (root_cells is not None) and (not isinstance(root_cells, str)):
        raise TypeError("root_cells must be a single CMB label (str) or None.")
    if (end_cells is not None) and (not isinstance(end_cells, str)):
        raise TypeError("end_cells must be a single CMB label (str) or None.")

    #### Validate plot_conn_graph_by
    valid_conn_options = {None, "Heatmap", "Network", "heatmap", "network"}
    if plot_conn_graph_by not in valid_conn_options:
        raise ValueError(
            f"plot_conn_graph_by must be None, 'Heatmap', or 'Network', got:  {plot_conn_graph_by}"
        )
    # Normalize to capitalized form
    if plot_conn_graph_by is not None:
        plot_conn_graph_by = plot_conn_graph_by.capitalize()

    required_cols = ["x_center", "y_center", "n_cells"]
    missing = [c for c in required_cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing columns in adata.obs: {missing}. Expected: {required_cols}")

    def _resolve_group_colors(unique_groups, user_colors, fallback_cmap="tab20"):
        """Build a color map {group -> color} from different user input types."""
        unique_groups = [str(g).strip() for g in unique_groups]
        n_groups = max(len(unique_groups), 1)
        default_cols = plt.cm.get_cmap(fallback_cmap)(np.linspace(0, 1, n_groups))
        default_map = dict(zip(unique_groups, default_cols))

        if user_colors is None:
            return default_map

        if isinstance(user_colors, dict):
            normalized = {str(k).strip(): v for k, v in user_colors.items()}
            return {g: normalized. get(g, default_map[g]) for g in unique_groups}

        if isinstance(user_colors, (list, tuple, np.ndarray)):
            seq = list(user_colors)
            if len(seq) == 0:
                return default_map
            colors = [seq[i % len(seq)] for i in range(len(unique_groups))]
            return dict(zip(unique_groups, colors))

        if isinstance(user_colors, str):
            try:
                cm = plt.cm.get_cmap(user_colors)
                colors = cm(np.linspace(0, 1, len(unique_groups)))
                return dict(zip(unique_groups, colors))
            except Exception: 
                return default_map

        return default_map

    def _coor_to_xy_and_cellids(coor_in):
        
        if isinstance(coor_in, np.ndarray):
            if coor_in.ndim != 2 or coor_in.shape[1] < 2:
                raise ValueError(f"coor ndarray must be (n,>=2); got {coor_in.shape}")
            raise ValueError()

        if isinstance(coor_in, pd. DataFrame):
            df = coor_in. copy()
            cols_lower = [c.lower() for c in df.columns]

            if "x" in cols_lower and "y" in cols_lower: 
                xcol = df.columns[cols_lower.index("x")]
                ycol = df.columns[cols_lower.index("y")]
                xy = df[[xcol, ycol]]. rename(columns={xcol: "x", ycol: "y"})
            elif "coor_x" in cols_lower and "coor_y" in cols_lower:
                xcol = df. columns[cols_lower.index("coor_x")]
                ycol = df.columns[cols_lower.index("coor_y")]
                xy = df[[xcol, ycol]]. rename(columns={xcol: "x", ycol: "y"})
            else:
                xy = df. iloc[:, :2]. rename(columns={df.columns[0]: "x", df.columns[1]: "y"})

            if coor_cell_col is not None:
                if coor_cell_col not in df.columns:
                    raise KeyError(f"coor_cell_col='{coor_cell_col}' not found in coor columns.")
                cell_ids = df[coor_cell_col].astype(str).str.strip().to_numpy()
            else:
                cell_ids = df.index.astype(str).str.strip().to_numpy()

            xy = xy.reset_index(drop=True)
            return xy, cell_ids

        raise TypeError("coor must be a pandas DataFrame (recommended) or numpy array.")

    def _cellinfo_to_series(cell_info_in):

        if isinstance(cell_info_in, pd.Series):
            s = cell_info_in.copy()
        elif isinstance(cell_info_in, pd.DataFrame):
            if cell_group_col in cell_info_in.columns:
                s = cell_info_in[cell_group_col].copy()
            elif cell_info_in.shape[1] == 1:
                s = cell_info_in.iloc[:, 0].copy()
            else:
                raise ValueError(
                    f"cell_info is a DataFrame but has no '{cell_group_col}' column and "
                    f"has {cell_info_in.shape[1]} columns; cannot infer group column."
                )
        else:
            raise TypeError("cell_info must be a pandas Series or DataFrame indexed by cell IDs.")

        s. index = s.index.astype(str).str.strip()
        s = s.astype(str).str.strip()
        return s

    def _compute_vector_directions(conn_matrix, labels, root_cells_local, end_cells_local):

        n = len(labels)
        W = conn_matrix. copy()

        label_to_idx = {lb: i for i, lb in enumerate(labels)}

        dist = None
        if root_cells_local is not None or end_cells_local is not None:
            target = root_cells_local if root_cells_local is not None else end_cells_local
            target = str(target).strip()

            if target not in label_to_idx: 
                return {}  

            target_idx = label_to_idx[target]

            rr, cc = np.where(W > 0)
            if rr.size == 0:
                dist = np.full(n, np.inf)
            else:
                eps = 1e-12
                lengths = 1.0 / (W[rr, cc] + eps)
                G = sp.csr_matrix((lengths, (rr, cc)), shape=(n, n))
                G = (G + G.T) * 0.5
                dist = dijkstra(G, directed=False, indices=target_idx)

        edge_directions = {}
        tol = 1e-12

        for i in range(n):
            for j in range(i + 1, n):
                if W[i, j] <= 0:
                    continue

                if dist is None:

                    edge_directions[(i, j)] = 0
                else:

                    if not np.isfinite(dist[i]) or not np.isfinite(dist[j]):
                        edge_directions[(i, j)] = 0
                    elif root_cells_local is not None: 

                        if dist[j] > dist[i] + tol:
                            edge_directions[(i, j)] = 1  
                        elif dist[i] > dist[j] + tol:
                            edge_directions[(i, j)] = -1  
                        else:
                            edge_directions[(i, j)] = 0
                    else: 
                        if dist[j] < dist[i] - tol:
                            edge_directions[(i, j)] = 1  # i -> j
                        elif dist[i] < dist[j] - tol: 
                            edge_directions[(i, j)] = -1  # j -> i
                        else: 
                            edge_directions[(i, j)] = 0

        return edge_directions
    ### Heatmap
    def _plot_connectivity_heatmap(ax, conn_matrix, labels, sort_labels=True):

        from matplotlib.colors import LinearSegmentedColormap

        # Sort labels and reorder matrix if requested
        if sort_labels: 
            sorted_indices = np.argsort(labels)
            sorted_labels = labels[sorted_indices]
            sorted_conn_matrix = conn_matrix[np.ix_(sorted_indices, sorted_indices)]
        else:
            sorted_labels = labels
            sorted_conn_matrix = conn_matrix

        n = len(sorted_labels)
        ###ColorSet
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom", ["#FFFBDE", "#90D1CA", "#129990", "#096B68"], N=256
        )
        im = ax.imshow(sorted_conn_matrix, cmap=custom_cmap, aspect="auto", interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Connectivity", fontsize=9)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(sorted_labels, fontsize=7)
        ax.tick_params(axis="both", which="both", length=0)
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("gray")
            spine.set_linewidth(0.5)

        ax.set_xlabel("CMB", fontsize=9)
        ax.set_ylabel("CMB", fontsize=9)
        ax.set_title("CMB Connectivity Heatmap", fontsize=10)

    def _plot_connectivity_network(
        ax, conn_matrix, labels, color_map, node_sizes,
        edge_directions, base_edge_alpha=0.6, show_labels=False
    ):
        n = len(labels)
        if n == 0:
            ax.text(0.5, 0.5, "No centers to display", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("CMB Connectivity Network", fontsize=10)
            return

        try:
            import networkx as nx

            G = nx.Graph()
            G.add_nodes_from(range(n))

            for i in range(n):
                for j in range(i + 1, n):
                    w = conn_matrix[i, j]
                    if w > 0:
                        G.add_edge(i, j, weight=w)

            if G.number_of_edges() > 0:
                pos = nx.spring_layout(
                    G,
                    k=2.0 / np.sqrt(n) if n > 1 else 1.0,
                    iterations=100,
                    weight="weight",
                    seed=42
                )
            else:
                pos = nx.circular_layout(G)

        except ImportError:
            print("[plot_cmb_cent_vector] WARNING: networkx not installed.  Using circular layout.")
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pos = {i: (np. cos(angles[i]), np.sin(angles[i])) for i in range(n)}

        positions = np.array([pos[i] for i in range(n)])

        conn_flat = conn_matrix[np.triu_indices(n, k=1)]
        if len(conn_flat) > 0 and conn_flat.max() > 0:
            max_conn = conn_flat.max()
            min_conn = conn_flat[conn_flat > 0].min() if np.any(conn_flat > 0) else 0
        else:
            max_conn = 1
            min_conn = 0

        for i in range(n):
            for j in range(i + 1, n):
                w = conn_matrix[i,j]
                if w <= 0:
                    continue

                if max_conn > min_conn:
                    norm_w = (w - min_conn) / (max_conn - min_conn)
                else:
                    norm_w = 0.5

                line_width = 0.8 + 3.5 * norm_w
                alpha = base_edge_alpha * (0.4 + 0.6 * norm_w)

                direction = edge_directions.get((i,j), 0)

                pos_i = positions[i]
                pos_j = positions[j]

                if direction == 0:
                    
                    ax.plot(
                        [pos_i[0], pos_j[0]],
                        [pos_i[1], pos_j[1]],
                        color="#555555",
                        linewidth=line_width,
                        alpha=alpha,
                        zorder=1,
                    )
                else:

                    if direction == 1:
 
                        start, end = pos_i, pos_j
                        start_idx, end_idx = i, j
                    else: 

                        start, end = pos_j, pos_i
                        start_idx, end_idx = j, i

                    shrink_start = np.sqrt(node_sizes[start_idx]) * 0.015
                    shrink_end = np.sqrt(node_sizes[end_idx]) * 0.015

                    arrow = FancyArrowPatch(
                        posA=start,
                        posB=end,
                        arrowstyle="-|>",
                        mutation_scale=8 + 6 * norm_w,
                        color="#555555",
                        linewidth=line_width,
                        alpha=alpha,
                        shrinkA=shrink_start * 50,
                        shrinkB=shrink_end * 50,
                        zorder=1,
                    )
                    ax.add_patch(arrow)

        node_colors = [color_map. get(str(labels[i]).strip(), "#444444") for i in range(n)]

        # Scale node sizes for network plot
        if node_sizes. max() > 0:
            scaled_sizes = 100 + 400 * (node_sizes / node_sizes. max())
        else:
            scaled_sizes = np.full(n,300)

        # Draw nodes
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=scaled_sizes,
            c=node_colors,
            alpha=0.95,
            marker="o",
            edgecolors="k",
            linewidths=0.8,
            zorder=5,
        )

        if show_labels:
            for i, lb in enumerate(labels):
                ax.annotate(
                    str(lb),
                    xy=(positions[i, 0], positions[i,1]),
                    xytext=(5,5),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    zorder=6,
                )

        padding = 0.15
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        ax.set_xlim(
            positions[:, 0].min() - padding * max(x_range, 0.5),
            positions[:, 0].max() + padding * max(x_range, 0.5)
        )
        ax.set_ylim(
            positions[: , 1].min() - padding * max(y_range, 0.5),
            positions[: , 1].max() + padding * max(y_range, 0.5)
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("CMB Connectivity Network", fontsize=10)

    # ======================== DATA PREPARATION ========================
    xy_cells, cell_ids = _coor_to_xy_and_cellids(coor)
    group_series = _cellinfo_to_series(cell_info)

    groups = group_series.reindex(cell_ids)
    unmapped_mask = groups.isna()
    if int(unmapped_mask.sum()) > 0:
        print(
            f"[plot_cmb_cent_vector] WARNING: {int(unmapped_mask.sum())} cells in coor could not be found "
            f"in cell_info index. They will be colored '#999999'."
        )
    groups = groups.fillna("#UNMAPPED#").astype(str)

    cell_df = xy_cells.copy()
    cell_df["group"] = groups. to_numpy()

    unique_groups = sorted(pd.unique(cell_df["group"]).tolist())
    cluster_color_map = _resolve_group_colors(unique_groups, group_colors, fallback_cmap="tab20")
    cluster_color_map["#UNMAPPED#"] = "#999999"

    if isinstance(group_colors, dict):
        norm_keys = {str(k).strip() for k in group_colors.keys()}
        missing_keys = [g for g in unique_groups if (g not in norm_keys) and (g != "#UNMAPPED#")]
        if len(missing_keys) > 0:
            print(
                f"[plot_cmb_cent_vector] WARNING: {len(missing_keys)} groups not found in group_colors; "
                f"they will fall back to tab20. Examples:  {missing_keys[: 30]}"
            )

    obs = adata.obs.copy()
    mask = obs[required_cols].notna().all(axis=1)
    obs2 = obs. loc[mask, required_cols]. astype(float)

    center_labels = adata.obs_names[mask]. astype(str).str.strip().to_numpy()
    x_cent = obs2["x_center"].to_numpy()
    y_cent = obs2["y_center"]. to_numpy()
    n_cells_cent = obs2["n_cells"].to_numpy()
    centers_xy = np.column_stack([x_cent, y_cent])
    n_cent = len(center_labels)

    if size_by_ncells and n_cent > 0:
        sizes = np.clip(n_cells_cent, 1, None)
        denom = np.sqrt(sizes. max()) if sizes.max() > 0 else 1.0
        s_cent = 20 + 180 * (np.sqrt(sizes) / denom)
    else:
        s_cent = np.full(n_cent, 160.0, dtype=float)

    center_color_map = _resolve_group_colors(
        sorted(set(center_labels. tolist())), group_colors, fallback_cmap="tab20"
    )
    center_colors = [center_color_map. get(lb, "#444444") for lb in center_labels]

    # ======================== CONNECTIVITY MATRIX ========================
    conn_matrix = None
    idx_cent = None

    if show_vector or plot_conn_graph_by is not None:
        if "joint_connectivities" in adata.obsp: 
            A = adata.obsp["joint_connectivities"]
        elif "joint" in adata.uns and "connectivities_key" in adata.uns["joint"]:
            A = adata.obsp[adata.uns["joint"]["connectivities_key"]]
        else:
            raise KeyError("Cannot find joint connectivities in adata.obsp['joint_connectivities'].")

        if not sp.issparse(A):
            A = sp.csr_matrix(A)
        A = A.tocsr()

        idx_cent = np.where(mask. to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask))[0]
        A = A[idx_cent][:, idx_cent]

        if A.shape != (n_cent, n_cent):
            raise ValueError(
                f"joint_connectivities shape {A.shape} must match centers count ({n_cent},{n_cent})."
            )

        A = (A + A.T) * 0.5
        A. setdiag(0.0)
        A. eliminate_zeros()

        conn_matrix = A.toarray()

    # ======================== COMPUTE EDGE DIRECTIONS ========================
    edge_directions = {}
    if conn_matrix is not None and (root_cells is not None or end_cells is not None):
        edge_directions = _compute_vector_directions(
            conn_matrix, center_labels, root_cells, end_cells
        )

    fig, ax = plt.subplots(figsize=figsize)

    if show_cells:
        for g in sorted(pd.unique(cell_df["group"]).tolist()):
            sub = cell_df[cell_df["group"] == g]
            ax.scatter(
                sub["x"], sub["y"],
                c=[cluster_color_map[g]],
                s=cell_size,
                alpha=cell_alpha,
                linewidths=0,
                zorder=1,
            )

    ax.scatter(
        x_cent, y_cent,
        s=s_cent,
        c=center_colors,
        alpha=0.95,
        marker="o",
        edgecolors="k",
        linewidths=0.6,
        zorder=5,
    )

    if show_labels:
        for xx, yy, lb in zip(x_cent, y_cent, center_labels):
            ax.text(xx, yy, lb, fontsize=8, color="black", zorder=6)

    target_idx = None
    if show_vector and conn_matrix is not None:
        W = conn_matrix. copy()

        dist = None
        if root_cells is not None or end_cells is not None: 
            target = (root_cells if root_cells is not None else end_cells)
            target = str(target).strip()
            label_to_idx = {lb: i for i, lb in enumerate(center_labels)}
            if target not in label_to_idx: 
                raise KeyError(f"Target CMB '{target}' not found among center labels.")
            target_idx = label_to_idx[target]

            rr, cc = np.where(W > 0)
            if rr.size == 0:
                dist = np.full(n_cent, np.inf)
            else:
                eps = 1e-12
                lengths = 1.0 / (W[rr, cc] + eps)
                G = sp.csr_matrix((lengths, (rr, cc)), shape=(n_cent, n_cent))
                G = (G + G.T) * 0.5
                dist = dijkstra(G, directed=False, indices=target_idx)

        V = np.zeros((n_cent, 2), dtype=float)
        tol = 1e-12
        for i in range(n_cent):
            w = W[i]. copy()

            if dist is not None:
                if not np.isfinite(dist[i]):
                    w[: ] = 0.0
                else:
                    if root_cells is not None: 
                        keep = np.isfinite(dist) & (dist > dist[i] + tol)
                    else: 
                        keep = np.isfinite(dist) & (dist < dist[i] - tol)
                    w *= keep. astype(float)

            sw = w.sum()
            if sw > 0:
                disp = centers_xy - centers_xy[i]
                V[i] = (w[: , None] * disp).sum(axis=0) / sw

        span_x = float(np.ptp(cell_df["x"]. to_numpy())) if show_cells else float(np.ptp(x_cent))
        span_y = float(np.ptp(cell_df["y"].to_numpy())) if show_cells else float(np.ptp(y_cent))
        span = max(span_x, span_y, 1.0)

        mag = np.linalg.norm(V, axis=1)
        nz = mag > 0
        if np.any(nz):
            med = float(np.median(mag[nz]))
            target_len = vector_span_frac * span
            scale = (target_len / med) if med > 0 else 1.0
            Vp = V * scale
        else:
            Vp = V

        ax.quiver(
            x_cent, y_cent,
            Vp[:, 0], Vp[:, 1],
            angles="xy", scale_units="xy", scale=1,
            width=0.003, headwidth=3.5, headlength=4.5,
            color="black",
            zorder=7,
        )

        if target_idx is not None: 
            ax.scatter(
                [x_cent[target_idx]], [y_cent[target_idx]],
                s=float(s_cent[target_idx]) * 1.25,
                facecolors="none",
                edgecolors="red",
                linewidths=1.8,
                zorder=8,
            )

    if show_group_legend:
        groups_for_legend = [g for g in sorted(pd. unique(cell_df["group"]).tolist()) if g != "#UNMAPPED#"]
        handles = [mpatches.Patch(color=cluster_color_map[g], label=g) for g in groups_for_legend]
        ax.legend(handles=handles, title="Group", bbox_to_anchor=(1, 1), loc="upper left")

    ax.set_title(title)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    try:
        ax.set_aspect(aspect)
    except Exception: 
        pass

    plt.tight_layout()

    # ======================== CONNECTIVITY GRAPH ========================
    fig_conn, ax_conn = None, None

    if plot_conn_graph_by is not None and conn_matrix is not None: 
        fig_conn, ax_conn = plt.subplots(figsize=conn_graph_figsize)

        if plot_conn_graph_by == "Heatmap":
            _plot_connectivity_heatmap(
                ax_conn, conn_matrix, center_labels,
                sort_labels=heatmap_sort_labels
            )
        elif plot_conn_graph_by == "Network":
            _plot_connectivity_network(
                ax_conn, conn_matrix, center_labels, center_color_map,
                node_sizes=s_cent,
                edge_directions=edge_directions,
                base_edge_alpha=network_edge_alpha,
                show_labels=show_labels
            )

        plt.tight_layout()

    if savepath: 
        parent = os.path.dirname(savepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

        fig.savefig(
            savepath,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
            metadata={"Title": title},
        )

        if fig_conn is not None:
            base, ext = os.path.splitext(savepath)
            conn_savepath = f"{base}_connectivity{ext}"
            fig_conn.savefig(
                conn_savepath,
                dpi=dpi,
                bbox_inches=bbox_inches,
                transparent=transparent,
                metadata={"Title": f"{title} - Connectivity"},
            )

    if show:
        plt.show()
    else:
        plt.close(fig)
        if fig_conn is not None: 
            plt.close(fig_conn)

    if plot_conn_graph_by is not None:
        return (fig, ax), (fig_conn, ax_conn)
    else:
        return fig, ax


def plot_cmb_conn(
    adata,
    coor,
    cell_info,
    group_colors=None,
    cell_group_col="group",
    coor_cell_col=None,
    show_cells=True,
    show_network=True,
    cell_alpha=0.3,
    cell_size=20,
    size_by_ncells=True,
    show_labels=True,
    show_group_legend=False,
    figsize=(9, 6),
    aspect="auto",
    title="CMB Connectivities Network",
    savepath=None,
    dpi=300,
    transparent=False,
    bbox_inches="tight",
    show=True,
    plot_heatmap=False,
    heatmap_figsize=(5, 5),
    heatmap_sort_labels=True,
    network_edge_alpha=0.1,
    network_edge_color="#234C6A",
    edge_quantile=1.0,
    conn_key="joint_connectivities_sym",
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import scipy.sparse as sp

    if not (0.0 < float(edge_quantile) <= 1.0):
        raise ValueError("edge_quantile must be in (0, 1]. Use 1 to disable filtering.")

    required_cols = ["x_center", "y_center", "n_cells"]
    missing = [c for c in required_cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing columns in adata.obs: {missing}. Expected: {required_cols}")

    def _resolve_group_colors(unique_groups, user_colors, fallback_cmap="tab20"):
        unique_groups = [str(g).strip() for g in unique_groups]
        n_groups = max(len(unique_groups), 1)
        default_cols = plt.cm.get_cmap(fallback_cmap)(np.linspace(0, 1, n_groups))
        default_map = dict(zip(unique_groups, default_cols))

        if user_colors is None:
            return default_map

        if isinstance(user_colors, dict):
            normalized = {str(k).strip(): v for k, v in user_colors.items()}
            return {g: normalized.get(g, default_map[g]) for g in unique_groups}

        if isinstance(user_colors, (list, tuple, np.ndarray)):
            seq = list(user_colors)
            if len(seq) == 0:
                return default_map
            colors = [seq[i % len(seq)] for i in range(len(unique_groups))]
            return dict(zip(unique_groups, colors))

        if isinstance(user_colors, str):
            try:
                cm = plt.cm.get_cmap(user_colors)
                colors = cm(np.linspace(0, 1, len(unique_groups)))
                return dict(zip(unique_groups, colors))
            except Exception:
                return default_map

        return default_map

    def _coor_to_xy_and_cellids(coor_in):
        if isinstance(coor_in, np.ndarray):
            if coor_in.ndim != 2 or coor_in.shape[1] < 2:
                raise ValueError(f"coor ndarray must be (n,>=2); got {coor_in.shape}")
            raise ValueError("ndarray input requires cell IDs; please use DataFrame instead.")

        if isinstance(coor_in, pd.DataFrame):
            df = coor_in.copy()
            cols_lower = [c.lower() for c in df.columns]

            if "x" in cols_lower and "y" in cols_lower:
                xcol = df.columns[cols_lower.index("x")]
                ycol = df.columns[cols_lower.index("y")]
                xy = df[[xcol, ycol]].rename(columns={xcol: "x", ycol: "y"})
            elif "coor_x" in cols_lower and "coor_y" in cols_lower:
                xcol = df.columns[cols_lower.index("coor_x")]
                ycol = df.columns[cols_lower.index("coor_y")]
                xy = df[[xcol, ycol]].rename(columns={xcol: "x", ycol: "y"})
            else:
                xy = df.iloc[:, :2].rename(columns={df.columns[0]: "x", df.columns[1]: "y"})

            if coor_cell_col is not None:
                if coor_cell_col not in df.columns:
                    raise KeyError(f"coor_cell_col='{coor_cell_col}' not found in coor columns.")
                cell_ids = df[coor_cell_col].astype(str).str.strip().to_numpy()
            else:
                cell_ids = df.index.astype(str).str.strip().to_numpy()

            xy = xy.reset_index(drop=True)
            return xy, cell_ids

        raise TypeError("coor must be a pandas DataFrame (recommended) or numpy array.")

    def _cellinfo_to_series(cell_info_in):
        if isinstance(cell_info_in, pd.Series):
            s = cell_info_in.copy()
        elif isinstance(cell_info_in, pd.DataFrame):
            if cell_group_col in cell_info_in.columns:
                s = cell_info_in[cell_group_col].copy()
            elif cell_info_in.shape[1] == 1:
                s = cell_info_in.iloc[:, 0].copy()
            else:
                raise ValueError(
                    f"cell_info is a DataFrame but has no '{cell_group_col}' column and "
                    f"has {cell_info_in.shape[1]} columns; cannot infer group column."
                )
        else:
            raise TypeError("cell_info must be a pandas Series or DataFrame indexed by cell IDs.")

        s.index = s.index.astype(str).str.strip()
        s = s.astype(str).str.strip()
        return s

    def _get_connectivity_matrix(adata_in, key, mask=None):
        if key not in adata_in.obsp:
            return None

        A = adata_in.obsp[key]
        if not sp.issparse(A):
            A = sp.csr_matrix(np.asarray(A))
        A = A.tocsr()

        if mask is not None:
            idx = np.where(mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask))[0]
            A = A[idx][:, idx]

        return A.toarray()

    def _plot_connectivity_heatmap(ax, conn_matrix, labels, sort_labels=True, title_suffix=""):
        from matplotlib.colors import LinearSegmentedColormap

        labels = np.asarray(labels)
        if sort_labels:
            sorted_indices = np.argsort(labels)
            sorted_labels = labels[sorted_indices]
            sorted_conn = conn_matrix[np.ix_(sorted_indices, sorted_indices)]
        else:
            sorted_labels = labels
            sorted_conn = conn_matrix

        sorted_conn = np.asarray(sorted_conn, dtype=float)
        n = len(sorted_labels)

        custom_cmap = LinearSegmentedColormap.from_list(
            "custom", ["#FFFBDE", "#90D1CA", "#129990", "#096B68"], N=256
        )
        im = ax.imshow(sorted_conn, cmap=custom_cmap, aspect="auto", interpolation="nearest")

        for i in range(n):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="#d6d6d6", zorder=10))

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Connectivity", fontsize=9)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(sorted_labels, fontsize=7)
        ax.tick_params(axis="both", which="both", length=0)
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("gray")
            spine.set_linewidth(0.5)

        ax.set_xlabel("CMB", fontsize=9)
        ax.set_ylabel("CMB", fontsize=9)
        ax.set_title(f"CMB Connectivity Heatmap{title_suffix}", fontsize=10)

    def _overlay_network_on_spatial(
        ax,
        conn_matrix,
        centers_xy,
        base_edge_alpha=0.1,
        edge_color="#555555",
        edge_quantile_local=1.0,
    ):
        n = conn_matrix.shape[0]
        if n == 0:
            return

        upper = conn_matrix[np.triu_indices(n, k=1)]
        pos = upper[upper > 0]
        if pos.size == 0:
            return

        thr = float(np.quantile(pos, edge_quantile_local)) if edge_quantile_local < 1.0 else -np.inf
        max_conn = float(pos.max())
        min_conn = float(pos.min()) if pos.size else 0.0

        for i in range(n):
            for j in range(i + 1, n):
                w = float(conn_matrix[i, j])
                if w <= 0 or w < thr:
                    continue

                if max_conn > min_conn:
                    norm_w = (w - min_conn) / (max_conn - min_conn)
                else:
                    norm_w = 0.5

                line_width = 0.1 + 2.0 * norm_w
                alpha = base_edge_alpha * (0.1 + 9.0 * norm_w)

                pos_i = centers_xy[i]
                pos_j = centers_xy[j]

                ax.plot(
                    [pos_i[0], pos_j[0]],
                    [pos_i[1], pos_j[1]],
                    color=edge_color,
                    linewidth=line_width,
                    alpha=alpha,
                    zorder=4,
                )

    def _create_spatial_plot_with_network(
        cell_df,
        cluster_color_map,
        centers_xy,
        center_labels,
        center_colors,
        x_cent,
        y_cent,
        s_cent,
        conn_matrix,
        show_cells_local,
        cell_size_local,
        cell_alpha_local,
        show_labels_local,
        show_group_legend_local,
        show_network_local,
        network_edge_alpha_local,
        network_edge_color_local,
        edge_quantile_local,
        figsize_local,
        aspect_local,
        plot_title,
    ):
        fig, ax = plt.subplots(figsize=figsize_local)

        if show_cells_local:
            for g in sorted(pd.unique(cell_df["group"]).tolist()):
                sub = cell_df[cell_df["group"] == g]
                ax.scatter(
                    sub["x"],
                    sub["y"],
                    c=[cluster_color_map[g]],
                    s=cell_size_local,
                    alpha=cell_alpha_local,
                    linewidths=0,
                    zorder=1,
                )

        if show_network_local and conn_matrix is not None:
            _overlay_network_on_spatial(
                ax,
                conn_matrix,
                centers_xy,
                base_edge_alpha=network_edge_alpha_local,
                edge_color=network_edge_color_local,
                edge_quantile_local=edge_quantile_local,
            )

        ax.scatter(
            x_cent,
            y_cent,
            s=s_cent,
            c=center_colors,
            alpha=0.95,
            marker="o",
            edgecolors="k",
            linewidths=0.6,
            zorder=5,
        )

        if show_labels_local:
            for xx, yy, lb in zip(x_cent, y_cent, center_labels):
                ax.text(xx, yy, lb, fontsize=8, color="black", zorder=6)

        if show_group_legend_local:
            groups_for_legend = [g for g in sorted(pd.unique(cell_df["group"]).tolist()) if g != "#UNMAPPED#"]
            handles = [mpatches.Patch(color=cluster_color_map[g], label=g) for g in groups_for_legend]
            ax.legend(handles=handles, title="Group", bbox_to_anchor=(1, 1), loc="upper left")

        ax.set_title(plot_title)
        ax.set_xlabel("Spatial X")
        ax.set_ylabel("Spatial Y")
        try:
            ax.set_aspect(aspect_local)
        except Exception:
            pass

        return fig, ax

    xy_cells, cell_ids = _coor_to_xy_and_cellids(coor)
    group_series = _cellinfo_to_series(cell_info)

    groups = group_series.reindex(cell_ids)
    unmapped_mask = groups.isna()
    if int(unmapped_mask.sum()) > 0:
        print(
            f"[plot_cmb_conn] WARNING: {int(unmapped_mask.sum())} cells in coor could not be found "
            f"in cell_info index. They will be colored '#999999'."
        )
    groups = groups.fillna("#UNMAPPED#").astype(str)

    cell_df = xy_cells.copy()
    cell_df["group"] = groups.to_numpy()

    unique_groups = sorted(pd.unique(cell_df["group"]).tolist())
    cluster_color_map = _resolve_group_colors(unique_groups, group_colors, fallback_cmap="tab20")
    cluster_color_map["#UNMAPPED#"] = "#999999"

    obs = adata.obs.copy()
    mask = obs[required_cols].notna().all(axis=1)
    obs2 = obs.loc[mask, required_cols].astype(float)

    center_labels = adata.obs_names[mask].astype(str).str.strip().to_numpy()
    x_cent = obs2["x_center"].to_numpy()
    y_cent = obs2["y_center"].to_numpy()
    n_cells_cent = obs2["n_cells"].to_numpy()
    centers_xy = np.column_stack([x_cent, y_cent])
    n_cent = len(center_labels)

    if size_by_ncells and n_cent > 0:
        sizes = np.clip(n_cells_cent, 1, None)
        denom = np.sqrt(sizes.max()) if sizes.max() > 0 else 1.0
        s_cent = 20 + 180 * (np.sqrt(sizes) / denom)
    else:
        s_cent = np.full(n_cent, 160.0, dtype=float)

    center_color_map = _resolve_group_colors(
        sorted(set(center_labels.tolist())), group_colors, fallback_cmap="tab20"
    )
    center_colors = [center_color_map.get(lb, "#444444") for lb in center_labels]

    conn_matrix = None
    if show_network or plot_heatmap:
        conn_matrix = _get_connectivity_matrix(adata, conn_key, mask)
        if conn_matrix is None:
            raise KeyError(
                f"Cannot find joint connectivity key '{conn_key}' in adata.obsp. "
                f"Available keys: {list(adata.obsp.keys())}"
            )
        if conn_matrix.shape != (n_cent, n_cent):
            raise ValueError(
                f"{conn_key} shape {conn_matrix.shape} must match centers count ({n_cent},{n_cent})."
            )

        conn_matrix = np.asarray(conn_matrix, dtype=float)
        np.fill_diagonal(conn_matrix, 0.0)

    fig, ax = _create_spatial_plot_with_network(
        cell_df,
        cluster_color_map,
        centers_xy,
        center_labels,
        center_colors,
        x_cent,
        y_cent,
        s_cent,
        conn_matrix,
        show_cells,
        cell_size,
        cell_alpha,
        show_labels,
        show_group_legend,
        show_network,
        network_edge_alpha,
        network_edge_color,
        edge_quantile,
        figsize,
        aspect,
        title,
    )
    plt.tight_layout()

    fig_heatmap, ax_heatmap = None, None
    if plot_heatmap and conn_matrix is not None:
        fig_heatmap, ax_heatmap = plt.subplots(figsize=heatmap_figsize)
        _plot_connectivity_heatmap(
            ax_heatmap,
            conn_matrix,
            center_labels,
            sort_labels=heatmap_sort_labels,
            title_suffix=f" | {conn_key}",
        )
        plt.tight_layout()

    if savepath:
        parent = os.path.dirname(savepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        base, ext = os.path.splitext(savepath)

        fig.savefig(
            savepath,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
            metadata={"Title": title},
        )

        if fig_heatmap is not None:
            fig_heatmap.savefig(
                f"{base}_heatmap{ext}",
                dpi=dpi,
                bbox_inches=bbox_inches,
                transparent=transparent,
            )

    if show:
        plt.show()
    else:
        plt.close(fig)
        if fig_heatmap is not None:
            plt.close(fig_heatmap)

    result = {"main": (fig, ax)}
    if fig_heatmap is not None:
        result["heatmap"] = (fig_heatmap, ax_heatmap)

    return result


def plot_cmb_vec(
    adata_cmb,
    coor_cell,
    cell_to_cmb,
    group_colors=None,
    coor_x="x",
    coor_y="y",
    cell_group_col="group",
    cmb_id_col=None,
    cmb_x_col="x_center",
    cmb_y_col="y_center",
    rank_key="cmb_snv_ratio_rank",
    conn_key="joint_connectivities",
    k_neighbors=100,  # number of neighbor cells used to determine if CMB is in spatial contact
    min_cmb_contact=5,  # filter contact
    use_contact_points=True,
    contact_sample_n=20,  # only affect arrow position
    contact_mode="cells_min",  # "cells_min", "cells_max" or "cells_sum"
    cell_size=20,
    cell_alpha=0.3,
    arrow_color="black",
    curvature=0.2,
    arrowstyle="-|>",
    arrow_mutation_scale=12,
    arrow_lw=1.6,
    arrow_min_alpha=0.05,
    arrow_max_alpha=1.0,
    draw_zero_weight=False,
    weight_quantiles=(0.05, 0.95),
    max_arrows=None,
    arrow_length_scale=None,
    arrow_min_length=None,
    figsize=(9, 6),
    aspect="auto",
    title="CMB Directional Vector Field",
    savepath=None,
    dpi=300,
    show=True,
    verbose=False,
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    from matplotlib.patches import FancyArrowPatch

    def _resolve_group_colors(unique_groups, user_colors, fallback_cmap="tab20"):
        unique_groups = [str(g).strip() for g in unique_groups]
        n = max(len(unique_groups), 1)
        default_cols = plt.cm.get_cmap(fallback_cmap)(np.linspace(0, 1, n))
        default_map = dict(zip(unique_groups, default_cols))
        if user_colors is None:
            return default_map
        if isinstance(user_colors, dict):
            user_colors = {str(k).strip(): v for k, v in user_colors.items()}
            return {g: user_colors.get(g, default_map[g]) for g in unique_groups}
        return default_map

    def _as_str_index(df):
        df = df.copy()
        df.index = df.index.astype(str)
        return df

    def _get_scalar(mat, i, j):
        v = mat[i, j]
        return float(v) if np.ndim(v) == 0 else float(np.asarray(v).ravel()[0])

    def _norm_weights(w_vals):
        w_vals = np.asarray(w_vals, dtype=float)
        if w_vals.size == 0:
            return w_vals
        if np.all(w_vals == w_vals[0]):
            return np.ones_like(w_vals)
        if np.all(w_vals == 0):
            return np.zeros_like(w_vals)
        if weight_quantiles is not None:
            qlo, qhi = weight_quantiles
            lo = np.quantile(w_vals, qlo) if w_vals.size >= 5 else float(w_vals.min())
            hi = np.quantile(w_vals, qhi) if w_vals.size >= 5 else float(w_vals.max())
        else:
            lo, hi = float(w_vals.min()), float(w_vals.max())
        if hi <= lo:
            return np.ones_like(w_vals)
        w_clip = np.clip(w_vals, lo, hi)
        return (w_clip - lo) / (hi - lo)

    def _contact_count(sa, sb, mode):
        na, nb = len(sa), len(sb)
        if mode == "cells_min":
            return min(na, nb)
        if mode == "cells_sum":
            return na + nb
        if mode == "cells_max":
            return max(na, nb)
        return min(na, nb)

    coor_cell = _as_str_index(coor_cell)
    cell_to_cmb = _as_str_index(cell_to_cmb)

    if coor_x not in coor_cell.columns or coor_y not in coor_cell.columns:
        raise KeyError(f"coor_cell must contain columns '{coor_x}' and '{coor_y}'")
    if cell_group_col not in cell_to_cmb.columns:
        raise KeyError(f"cell_to_cmb must contain column '{cell_group_col}'")

    common_cells = coor_cell.index.intersection(cell_to_cmb.index)
    if common_cells.size == 0:
        raise ValueError("No overlapping cell IDs between coor_cell.index and cell_to_cmb.index")

    xy = coor_cell.loc[common_cells, [coor_x, coor_y]].rename(columns={coor_x: "x", coor_y: "y"}).dropna()
    common_cells = xy.index
    cmb_of_cell = cell_to_cmb.loc[common_cells, cell_group_col].astype(str)

    if verbose:
        print(f"[cell] n_cells={len(common_cells)}, n_cmb_in_cells={cmb_of_cell.nunique()}")

    fig, ax = plt.subplots(figsize=figsize)

    unique_groups = sorted(cmb_of_cell.unique())
    color_map = _resolve_group_colors(unique_groups, group_colors)
    for g in unique_groups:
        idx = (cmb_of_cell == g).to_numpy()
        sub = xy.iloc[np.where(idx)[0]]
        ax.scatter(sub["x"], sub["y"], s=cell_size, alpha=cell_alpha, c=[color_map[g]], linewidths=0, zorder=1)

    if rank_key not in adata_cmb.obs:
        raise KeyError(f"{rank_key} not found in adata_cmb.obs")
    if conn_key not in adata_cmb.obsp:
        raise KeyError(f"{conn_key} not found in adata_cmb.obsp")

    if cmb_id_col is not None:
        if cmb_id_col not in adata_cmb.obs.columns:
            raise KeyError(f"cmb_id_col='{cmb_id_col}' not found in adata_cmb.obs")
        cmb_ids = adata_cmb.obs[cmb_id_col].astype(str).to_numpy()
    else:
        cmb_ids = adata_cmb.obs_names.astype(str).to_numpy()

    cmb_index = {cid: i for i, cid in enumerate(cmb_ids)}
    rank_series = adata_cmb.obs[rank_key].astype(float).to_numpy()
    rank_map = {cid: rank_series[i] for i, cid in enumerate(cmb_ids)}

    if (cmb_x_col in adata_cmb.obs.columns) and (cmb_y_col in adata_cmb.obs.columns):
        cx = adata_cmb.obs[cmb_x_col].astype(float).to_numpy()
        cy = adata_cmb.obs[cmb_y_col].astype(float).to_numpy()
        center_map = {cid: (cx[i], cy[i]) for i, cid in enumerate(cmb_ids)}
    else:
        tmp = xy.copy()
        tmp["cmb"] = cmb_of_cell.values
        cent = tmp.groupby("cmb")[["x", "y"]].mean()
        center_map = {cid: (float(cent.loc[cid, "x"]), float(cent.loc[cid, "y"])) for cid in cent.index}

    W = adata_cmb.obsp[conn_key]
    if not hasattr(W, "tocsr"):
        W = sp.csr_matrix(W)
    W = W.tocsr()

    xy_arr = xy[["x", "y"]].to_numpy()
    tree = cKDTree(xy_arr)
    _, nn = tree.query(xy_arr, k=min(k_neighbors + 1, xy_arr.shape[0]))

    cmb_arr = cmb_of_cell.to_numpy()
    pair_side_cells = {}
    pair_side_pts = {}

    n = xy_arr.shape[0]
    for i in range(n):
        ci = cmb_arr[i]
        for j in nn[i][1:]:
            cj = cmb_arr[j]
            if ci == cj:
                continue
            a, b = (ci, cj) if ci < cj else (cj, ci)
            key = (a, b)
            d = pair_side_cells.get(key)
            if d is None:
                d = {a: set(), b: set()}
                pair_side_cells[key] = d
            d[ci].add(i)
            d[cj].add(j)
            if use_contact_points:
                p = pair_side_pts.get(key)
                if p is None:
                    p = {a: set(), b: set()}
                    pair_side_pts[key] = p
                p[ci].add(i)
                p[cj].add(j)

    edges = []
    for (a, b), d in pair_side_cells.items():
        if a not in cmb_index or b not in cmb_index:
            continue
        cnt = _contact_count(d[a], d[b], contact_mode)
        if cnt < min_cmb_contact:
            continue
        ra, rb = rank_map.get(a, np.nan), rank_map.get(b, np.nan)
        if not np.isfinite(ra) or not np.isfinite(rb) or ra == rb:
            continue
        src, dst = (a, b) if ra < rb else (b, a)
        i, j = cmb_index[src], cmb_index[dst]
        w = max(_get_scalar(W, i, j), _get_scalar(W, j, i))
        if (w <= 0) and (not draw_zero_weight):
            continue
        edges.append((src, dst, float(w), (a, b), int(cnt)))

    if verbose:
        print(f"[edges] candidate_pairs={len(pair_side_cells)}, eligible_edges={len(edges)}")
        if len(edges) > 0:
            ws = np.array([e[2] for e in edges], float)
            cs = np.array([e[4] for e in edges], int)
            print(f"[edges] weight: min={ws.min():.3g}, median={np.median(ws):.3g}, max={ws.max():.3g}")
            print(f"[edges] contact: min={cs.min()}, median={np.median(cs)}, max={cs.max()}")

    if len(edges) == 0:
        ax.set_title(title + " (no eligible CMB-CMB arrows)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(aspect)
        plt.tight_layout()

        # --- FIX: safe save even if savepath has no directory ---
        if savepath is not None:
            savepath_abs = os.path.abspath(savepath)
            os.makedirs(os.path.dirname(savepath_abs), exist_ok=True)
            fig.savefig(savepath_abs, dpi=dpi, bbox_inches="tight")
        # --------------------------------------------------------

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax

    if max_arrows is not None and max_arrows > 0:
        edges = sorted(edges, key=lambda x: x[2], reverse=True)[:max_arrows]

    w_norm = _norm_weights([e[2] for e in edges])

    for (src, dst, w, undirected_key, cnt), wn in zip(edges, w_norm):
        aalpha = float(arrow_min_alpha) + float(wn) * (float(arrow_max_alpha) - float(arrow_min_alpha))
        if aalpha > 1.0:
            aalpha = 1.0
        if aalpha < 0.0:
            aalpha = 0.0

        if use_contact_points and undirected_key in pair_side_pts:
            p = pair_side_pts[undirected_key]
            src_idx = sorted(list(p.get(src, set())))
            dst_idx = sorted(list(p.get(dst, set())))
            if len(src_idx) > 0 and len(dst_idx) > 0:
                src_idx = src_idx[:contact_sample_n]
                dst_idx = dst_idx[:contact_sample_n]
                x0, y0 = xy_arr[src_idx].mean(axis=0)
                x1, y1 = xy_arr[dst_idx].mean(axis=0)
            else:
                x0, y0 = center_map.get(src, (np.nan, np.nan))
                x1, y1 = center_map.get(dst, (np.nan, np.nan))
        else:
            x0, y0 = center_map.get(src, (np.nan, np.nan))
            x1, y1 = center_map.get(dst, (np.nan, np.nan))

        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
            continue

        if arrow_length_scale is not None or arrow_min_length is not None:
            dx, dy = (x1 - x0), (y1 - y0)
            L = (dx * dx + dy * dy) ** 0.5
            if L > 0:
                s = 1.0 if arrow_length_scale is None else float(arrow_length_scale)
                if arrow_min_length is not None and L * s < float(arrow_min_length):
                    s = float(arrow_min_length) / L
                x1, y1 = x0 + s * dx, y0 + s * dy

        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle=arrowstyle,
                connectionstyle=f"arc3,rad={curvature}",
                linewidth=float(arrow_lw),
                color=arrow_color,
                alpha=aalpha,
                zorder=3,
                mutation_scale=arrow_mutation_scale,
            )
        )

    ax.set_title(title)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    ax.set_aspect(aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if savepath is not None:
        savepath_abs = os.path.abspath(savepath)
        os.makedirs(os.path.dirname(savepath_abs), exist_ok=True)
        fig.savefig(savepath_abs, dpi=dpi, bbox_inches="tight")
    # --------------------------------------------------------

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
