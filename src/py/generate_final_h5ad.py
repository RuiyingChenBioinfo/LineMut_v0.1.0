import os
import math
from os import path
from glob import glob
from functools import partial
from multiprocessing import Pool
from collections import defaultdict

import vcf
import pysam
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
from pysam import AlignmentFile as AFile

from intersect_vcf import SNV


def read_vcf(vcf_fp: str) -> SNV:
    f = vcf.Reader(filename=vcf_fp)
    for rcd in f:
        if rcd.is_snp:
            for alt in rcd.ALT:
                yield SNV(rcd.CHROM, rcd.POS, rcd.REF, alt.sequence)


def get_snv_set(vcf_fp: str) -> set[SNV]:
    f = vcf.Reader(filename=vcf_fp)
    snvs = set()
    for rcd in f:
        if rcd.is_snp:
            snvs.update(
                SNV(rcd.CHROM, rcd.POS, rcd.REF, alt.sequence)
                for alt in rcd.ALT
            )
    return snvs


def unit_proc(
    unit_bam: str,
    unit_barcode_lst: list[str],
    snv_collection: dict[SNV, dict],
    locus2snvs: dict[str, dict],
    cb_tag: str,
    r_end_clip: int,
    support_threshold: int,
    max_read_len: int=150,
    max_gap: int=1000
) -> tuple[pd.Series, pd.Series]:

    flank: int = math.ceil(max_read_len * 1.5)  # pileup 稳定窗口大小

    def _windows(sorted_pos: list[int]):
        if not sorted_pos: return
        start = sorted_pos[0]
        end = sorted_pos[0]
        targets = [sorted_pos[0],]
        for p in sorted_pos[1:]:
            if p - end <= max_gap:
                end = p
                targets.append(p)
            else:
                # 在窗口两端添加额外的稳定窗口，保证两端位点pileup结果的可靠
                yield (max(start - 1 - flank, 0), end + flank, set(targets))
                start = p
                end = p
                targets = [p,]
        # 在窗口两端添加额外的稳定窗口，保证两端位点pileup结果的可靠
        yield (max(start - 1 - flank, 0), end + flank, set(targets))

    def _pileup_col_proc(pileup_col, alt):
        depth, support = defaultdict(set), defaultdict(set)
        for pileup_read in pileup_col.pileups:
            if (pos := pileup_read.query_position) is not None:
                seg = pileup_read.alignment
                umi = seg.get_tag("UB") if seg.has_tag("UB") else "NA"
                cb = seg.get_tag(cb_tag)
                depth[cb].add(umi)
                read_seq = seg.query_sequence
                if read_seq[pos].upper() == alt:
                    if pos >= r_end_clip and pos < (len(read_seq) - r_end_clip):
                        support[cb].add(umi)
        return depth, support

    label = path.basename(unit_bam).replace(".bam", '')
    snv_str_lst, unit_ratio_lst, unit_depth_lst = [], [], []
    sc_ratio_dict, sc_depth_dict = defaultdict(dict), defaultdict(dict)
    with AFile(unit_bam, "rb") as unit_bam:
        for chrom, pos_snv_map in locus2snvs.items():
            for start, end, targets in _windows(sorted(pos_snv_map.keys())):
                for pileup_col in unit_bam.pileup(
                    chrom, start, end, truncate=True,
                    stepper="nofilter", max_depth=1000000
                ):
                    if (rpos := pileup_col.reference_pos + 1) not in targets:
                        continue
                    for snv in pos_snv_map[rpos]:
                        depth, support = _pileup_col_proc(pileup_col, snv.alt)
                        total_depth = sum(len(s) for s in depth.values())
                        total_support = sum(len(s) for s in support.values())
                        ratio = 0
                        info = snv_collection[snv]
                        if label in info:
                            if info[label] or total_support >= support_threshold:
                                if total_depth > 0:
                                    ratio = total_support / total_depth
                        snv_str = str(snv)
                        snv_str_lst.append(snv_str)
                        unit_ratio_lst.append(ratio)
                        unit_depth_lst.append(total_depth)
                        for cb in depth.keys() | support.keys():
                            sc_depth = len(depth[cb])
                            sc_support = len(support[cb])
                            sc_ratio = sc_support / sc_depth if sc_depth > 0 else 0
                            sc_depth_dict[cb][snv_str] = sc_depth
                            sc_ratio_dict[cb][snv_str] = sc_ratio

    sc_depth_dict = {cb : pd.Series(d, dtype="uint32") for cb, d in sc_depth_dict.items()}
    sc_ratio_dict = {cb : pd.Series(d, dtype="float32") for cb, d in sc_ratio_dict.items()}

    return (pd.Series(unit_ratio_lst, index=snv_str_lst, dtype="float32"),
            pd.Series(unit_depth_lst, index=snv_str_lst, dtype="uint32"),
            pd.DataFrame(sc_depth_dict, index=snv_str_lst,
                         columns=unit_barcode_lst).fillna(0),
            pd.DataFrame(sc_ratio_dict, index=snv_str_lst,
                         columns=unit_barcode_lst).fillna(0))


def get_unit_anndata(
    label_lst: list[str],
    depth_ser_lst: list[pd.Series],
    ratio_ser_lst: list[pd.Series]
) -> ad.AnnData:

    def construct_anndata(ser_lst):
        d = {label : ser for label, ser in zip(label_lst, ser_lst)}
        df = pd.DataFrame(d).fillna(0).T
        adata = ad.AnnData(csr_matrix(df.to_numpy()))
        adata.obs_names, adata.var_names = df.index, df.columns
        return adata

    unit_depth_adata = construct_anndata(depth_ser_lst)
    unit_ratio_adata = construct_anndata(ratio_ser_lst)
    unit_ratio_adata = unit_ratio_adata[:, unit_ratio_adata.X.sum(axis=0) != 0]
    unit_depth_adata = unit_depth_adata[:, unit_ratio_adata.var_names]
    return unit_depth_adata, unit_ratio_adata


def df2adata(df_lst: list[pd.DataFrame]) -> ad.AnnData:
    df = pd.concat(df_lst, axis=1).fillna(0).T
    adata = ad.AnnData(csr_matrix(df.to_numpy()))
    adata.obs_names, adata.var_names = df.index, df.columns
    return adata


def get_snv_collection(
    trusted_vcf_lst: list[str],
    untrusted_vcf_lst: list[str],
    bam_lst: list[str]
) -> dict[SNV, dict]:

    snv_collection = defaultdict(dict)
    for tvcf, uvcf, bam in zip(trusted_vcf_lst, untrusted_vcf_lst, bam_lst):
        label = path.basename(bam).replace(".bam", '')
        for s in read_vcf(tvcf):
            snv_collection[s][label] = True
        for s in read_vcf(uvcf):
            snv_collection[s][label] = False
    return snv_collection


def main(
    categorized_vars_dir: str,
    split_bams_dir: str,
    output_dir: str,
    cell_barcode_tag: str,
    barcode_unit_map_file: str,
    reads_end_clip: int=5,
    support_threshold: int=5
) -> None:

    trusted_vcfs = glob(
        path.join(categorized_vars_dir, "trusted/**/*.vcf"), recursive=True
    )
    unit_labels = [
        path.basename(fp).replace("_trusted_sorted.vcf", '')
        for fp in trusted_vcfs
    ]
    untrusted_vcfs = [
        path.join(
            categorized_vars_dir,
            f"untrusted/{label}/{label}_untrusted_sorted.vcf"
        ) for label in unit_labels
    ]
    bams = [path.join(split_bams_dir, f"{label}.bam") for label in unit_labels]
    bc_unit_map = pd.read_csv(barcode_unit_map_file, names=["barcode", "unit"])

    def task_args():
        for label, bam in zip(unit_labels, bams):
            bcs = bc_unit_map[bc_unit_map["unit"] == label]["barcode"].tolist()
            yield bam, bcs

    snv_collection = get_snv_collection(trusted_vcfs, untrusted_vcfs, bams)
    # 废弃，因为多进程传参时，lambda函数对象无法序列化
    # locus2snvs = defaultdict(lambda : defaultdict(list))
    locus2snvs = dict()
    for snv in snv_collection.keys():
        d = locus2snvs.setdefault(snv.chrom, dict())
        d.setdefault(snv.pos, list()).append(snv)
        # locus2snvs[snv.chrom][snv.pos].append(snv)
    partial_unit_proc = partial(
        unit_proc,
        snv_collection=snv_collection,
        locus2snvs=locus2snvs,
        cb_tag=cell_barcode_tag,
        r_end_clip=reads_end_clip,
        support_threshold=support_threshold
    )
    nproc = min(os.cpu_count(), len(unit_labels))
    with Pool(nproc) as pp:
        chunksize = len(unit_labels) // nproc
        ret_val = pp.starmap(
            partial_unit_proc, task_args(), chunksize=chunksize
        )
    ratio_ser_lst, depth_ser_lst, sc_depth_lst, sc_ratio_lst = zip(*ret_val)
    unit_depth_adata, unit_ratio_adata = get_unit_anndata(
        unit_labels, depth_ser_lst, ratio_ser_lst
    )
    unit_ratio_adata.write_h5ad(path.join(output_dir, "unit_ratio.h5ad"))
    unit_depth_adata.write_h5ad(path.join(output_dir, "unit_depth.h5ad"))
    del depth_ser_lst, ratio_ser_lst

    sc_depth_adata = df2adata(sc_depth_lst)[:, unit_ratio_adata.var_names]
    sc_ratio_adata = df2adata(sc_ratio_lst)[:, unit_ratio_adata.var_names]
    sc_ratio_adata.write_h5ad(path.join(output_dir, "sc_ratio.h5ad"))
    sc_depth_adata.write_h5ad(path.join(output_dir, "sc_depth.h5ad"))


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])

