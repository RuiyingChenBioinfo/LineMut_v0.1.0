import os
import sys
import time
from os import path
from glob import glob
from typing import Union
from functools import partial
from multiprocessing import Pool
from multiprocessing import Event
from multiprocessing import Process
from collections import OrderedDict
from collections import defaultdict

import psutil
import leidenalg
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from pysam import AlignmentFile as AFile
from sklearn.cluster import SpectralClustering
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer


def construct_kmer_anndata(
    input_bam_pathname: str,
    *,
    k: int=9,
    cell_barcode_tag: str="CB",
) -> AnnData:

    current_kmer_index = -1
    kmer_index_map = OrderedDict()
    cells_kmers_counter = OrderedDict()
    with AFile(input_bam_pathname, "rb") as bam:
        for segment in bam:
            barcode = segment.get_tag(cell_barcode_tag)
            sequence = segment.query_sequence.upper()
            if (seq_len := len(sequence)) >= k:
                for i in range(seq_len - k + 1):
                    kmer = sequence[i : i + k]
                    if kmer in kmer_index_map:
                        kmer_index = kmer_index_map[kmer]
                    else:
                        current_kmer_index += 1
                        kmer_index = current_kmer_index
                        kmer_index_map[kmer] = kmer_index
                    cells_kmers_counter.setdefault(
                        barcode, defaultdict(int)
                    )[kmer_index] += 1

    data, row_ind, col_ind = [], [], []
    for cell_idx, (_, counter) in enumerate(cells_kmers_counter.items()):
        for kmer_idx, count in counter.items():
            data.append(count)
            row_ind.append(cell_idx)
            col_ind.append(kmer_idx)

    kmer_adata = AnnData(
        csr_matrix(
            (data, (row_ind, col_ind)),
            dtype=np.int64,
            shape=(len(cells_kmers_counter), len(kmer_index_map))
        )
    )
    kmer_adata.obs_names = list(cells_kmers_counter.keys())
    kmer_adata.var_names = list(kmer_index_map.keys())

    return kmer_adata


def kmer_mtx_dim_reduction(kmer_adata: AnnData):
    X_tfidf = TfidfTransformer().fit_transform(kmer_adata.X)
    if (min_dim := min(kmer_adata.shape)) > 100:
        n = 100
    else:
        n = min_dim - 1
    tsvd = TruncatedSVD(n_components=n, algorithm='arpack')
    kmer_adata.obsm["reduced_kmer_mtx"] = tsvd.fit_transform(X_tfidf)
    return kmer_adata


def find_optimal_cluster_number(
    cells_similarity_mtx: np.ndarray,
    # kmer_reduced_mtx: np.ndarray,
    max_clusters: int
) -> tuple[int, int]:

    min_clusters = 2
    max_clusters = min(30, max_clusters)
    cluster_labels_list, dbscores = [], []
    np.fill_diagonal(cells_similarity_mtx, 0)
    for k in range(min_clusters, max_clusters + 1):
        cluster_labels = SpectralClustering(
            n_clusters=k, affinity="precomputed"
        ).fit_predict(cells_similarity_mtx)
        # dbscore = davies_bouldin_score(kmer_reduced_mtx, cluster_labels)
        dbscore = davies_bouldin_score(cells_similarity_mtx, cluster_labels)
        dbscores.append(dbscore)

    # best_k = (biggest_score_idx := np.argmax(dbscores)) + min_clusters
    sorted_score_idx = np.argsort(dbscores)
    if len(dbscores) > 3:
        smallest_score_idx = max(sorted_score_idx[:3])
    else:
        smallest_score_idx = sorted_score_idx[0]
    best_k = smallest_score_idx + min_clusters
    best_score = dbscores[smallest_score_idx]
    return best_k, best_score


def assign_cellmetabin(
    kmer_adata: AnnData,
    cells_coor_csv_file: str,
    distance_weight: float
) -> AnnData:

    kmer_cosine_similarity_mtx = (
        cosine_similarity(kmer_adata.obsm["reduced_kmer_mtx"]) + 1
    ) / 2
    n_cells = kmer_adata.n_obs
    cells_coor_mtx = pd.read_csv(
        cells_coor_csv_file,
        names=["barcode", "coor_x", "coor_y"],
        index_col="barcode"
    ).reindex(kmer_adata.obs_names)
    cells_distance_mtx = euclidean_distances(cells_coor_mtx.to_numpy())
    maximum, minimum = cells_distance_mtx.max(), cells_distance_mtx.min()
    cells_distance_mtx = (
        1 - ((cells_distance_mtx - minimum) / (maximum - minimum))
    )

    cells_simi_mtx = 0.5 * kmer_cosine_similarity_mtx + 0.5 * cells_distance_mtx
    kmer_adata.obsm["cells_similarity_mtx"] = cells_simi_mtx

    optimal_cluster_number, dbscore = find_optimal_cluster_number(
        cells_simi_mtx, n_cells // 5
    )

    kmer_adata.obs["cellmetabin"] = SpectralClustering(
        n_clusters=optimal_cluster_number, affinity='precomputed'
    ).fit_predict(cells_simi_mtx)

    return kmer_adata, optimal_cluster_number, dbscore


def single_bam_proc(
    input_bam_filepath: str,
    *,
    k: int,
    output_dirpath: str,
    cells_coordinate_csvfile: str, 
    cell_barcode_tag: str
) -> None:

    celltype = path.basename(input_bam_filepath).replace(".bam", '')
    kmer_adata = construct_kmer_anndata(
        input_bam_filepath, k=k, cell_barcode_tag=cell_barcode_tag
    )
    if (n_cell := kmer_adata.n_obs) < 10:
        barcodes = kmer_adata.obs_names
        return pd.Series([f"{celltype}_cmb_0"] * n_cell, index=barcodes), 1, 0
    kmer_adata = kmer_mtx_dim_reduction(kmer_adata)
    kmer_adata, optimal_cluster_number, optimal_dbscore = assign_cellmetabin(
        kmer_adata, cells_coordinate_csvfile, 0.5
    )
    input_bam_filename = path.basename(input_bam_filepath).replace(".bam", '')
    kmer_adata.write_h5ad(
        path.join(output_dirpath, f"{input_bam_filename}_{k}_mer.h5ad")
    )
    return (
        f"{celltype}_cmb_" + kmer_adata.obs["cellmetabin"].copy().astype(str),
        optimal_cluster_number,
        optimal_dbscore
    )

def memory_monitor(
    ppid: int,
    profile_output_dir: str,
    k: int,
    start: Event,
    time_interval: int=5
) -> None:
    """
    监控内存占用情况
    """
    parent = psutil.Process(ppid)
    memory_usage = list()
    subprocess_started = False
    start.wait()
    while start.is_set():
        total_mem_usage = sum(
            p.memory_info().rss for p in parent.children() + [parent,]
        )
        memory_usage.append(total_mem_usage / 1024 / 1024)
        time.sleep(time_interval)
    with open(path.join(profile_output_dir, f"{k}_mer_mem_usage.txt"), "wt") as f:
        f.write('\n'.join(f"{v:.2f}" for v in memory_usage))


def main(
    split_bams_dirpath: str,
    k: str,
    output_dirpath: str,
    cells_coordinate_csv_file: str,
    cell_barcode_tag: str="CB", 
    profile_output_dir: Union[None, str]=None
) -> None:

    # output_dirpath = path.join(output_dirpath, f"{k}_mer")
    # os.makedirs(output_dirpath)

    input_bam_files = glob(path.join(split_bams_dirpath, "*.bam"))
    func = partial(
        single_bam_proc,
        k=int(k),
        output_dirpath=output_dirpath,
        cell_barcode_tag=cell_barcode_tag, 
        cells_coordinate_csvfile=cells_coordinate_csv_file
    )

    if profile_output_dir:
        start = Event()
        monitor_p = Process(
            target=memory_monitor,
            args=(os.getpid(), profile_output_dir, int(k), start)
        )
        monitor_p.start()
    else:
        start, monitor_p = None, None

    with Pool(min(len(input_bam_files), os.cpu_count() // 2 + 1)) as pp:
        if start is not None: start.set()
        returned_values = pp.map(func, input_bam_files)
        returned_series, returned_cluster_numbers, returned_dbscores = zip(
            *returned_values
        )
    pd.concat(returned_series).to_csv(
        path.join(output_dirpath, "cellmetabin.csv"), header=False
    )

    if (start is not None) and (monitor_p is not None):
        start.clear()
        monitor_p.join()

    if profile_output_dir:
        f = open(path.join(profile_output_dir, f"{k}_mer_cluster_info.csv"), "wt")
        f.write("optimal_cluster_numbers,corresponding_dbscore\n")
        for num, score in zip(returned_cluster_numbers, returned_dbscores):
            f.write(f"{num},{score}\n")
        f.close()


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
