import re
import os
import sys
from os import path
from typing import Union
from typing import Hashable

import pysam
from pysam import AlignmentFile as AFile


def filter_split_bam(
    raw_bam_pathname: str,
    split_bams_dir: str,
    *,
    barcode_group_map: dict[str, Hashable],
    cell_barcode_tag: str="CB",
    filter_out_saveas: Union[str, None]=None,
) -> None:

    """
    过滤 BAM 文件中的 reads 并拆分保存到不同子 BAM 中

    过滤 1) 不在给定 barcode 列表中的 reads；2) 没有参考基因组比对位置信息的
    reads，即没有比对到参考基因组上的reads; 3) 非主要比对及 duplicated reads
    将最终过滤之后的 reads 分组保存到不同的 BAM 文件中

    参数:
    -----
        raw_bam_pathname: str
            原始 BAM 文件路径名
        split_bams_dir: str
            保存过滤之后，拆分的子 BAM 的目录，因此需要保证拥有该目录的写入权限，
            目录可以是一个已存在的空目录，也可以是一个不存在的目录。如果是不存在
            的目录则自动创建
        barcode_group_map: dict[str, Hashable]
            细胞 barcode 与对应所属类别（如 cluster，celltype 等）的字典
        cell_barcode_tag: str
            barcode_group_map 中细胞 barcode 在原始 BAM 中对应的 tag，默认 'CB'
        filter_out_saveas: str | None
            过滤掉的 reads 的保存文件路径名，默认为 None，不单独保存这些 reads
    """

    if path.isdir(split_bams_dir):
        if len(os.listdir(split_bams_dir)) > 0:
            raise ValueError(f"'{split_bams_dir}' exists and is not empty.")
    else:
        os.makedirs(split_bams_dir)

    # 去除 FLAG 含 4，256，512，1024，2048 的 reads
    flag_mask = 0x0004 | 0x0100 | 0x0200 | 0x0400 | 0x0800

    def _check_reads(segment: pysam.AlignedSegment) -> bool:
        """
        检查 reads 是否保留
        """
        if (segment.reference_name is None) or (-1 == segment.reference_start):
            return False
        if segment.flag & flag_mask:
            return False
        if segment.has_tag(cell_barcode_tag):
            if segment.get_tag(cell_barcode_tag) not in barcode_group_map:
                return False
        else:
            return False
        return True

    join = path.join
    sub_chars = re.compile(r"\W+", re.A)
    with AFile(raw_bam_pathname, "rb") as raw_bam:
        new_bams = {grp : join(split_bams_dir, sub_chars.sub('_', grp) + ".bam")
                    for grp in set(barcode_group_map.values())}
        new_bams = {grp : AFile(pathname, "wb", template=raw_bam)
                    for grp, pathname in new_bams.items()}
        filter_out_bam = AFile(
            filter_out_saveas if filter_out_saveas else os.devnull,
            "wb",
            template=raw_bam
        )
        try:
            for segment in raw_bam:
                if _check_reads(segment):
                    group = barcode_group_map[segment.get_tag(cell_barcode_tag)]
                    new_bams[group].write(segment)
                else:
                    filter_out_bam.write(segment)
        finally:
            for f in new_bams.values(): f.close()
            filter_out_bam.close()


def main(
    raw_bam_filepath: str,
    split_bams_output_dirpath: str,
    barcode_celltype_csv_filepath: str,
    cell_barcode_tag: str="CB", 
    filter_out_saveas: Union[str, None]=None,
) -> None:

    barcode_celltype_map = dict()
    with open(barcode_celltype_csv_filepath, "rt") as f:
        for line in filter(None, map(str.strip, f)):
            barcode, celltype = line.split(',', 1)
            barcode_celltype_map[barcode] = celltype
    filter_split_bam(
        raw_bam_filepath, split_bams_output_dirpath,
        barcode_group_map=barcode_celltype_map,
        cell_barcode_tag=cell_barcode_tag,
        filter_out_saveas=filter_out_saveas
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
