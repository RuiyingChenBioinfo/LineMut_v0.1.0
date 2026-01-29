import os
from os import path
from glob import glob
from functools import partial
from multiprocessing import Pool

from pysam import AlignmentFile as AFile


def split_bam(
    bam_file: str, 
    cmb_dict: dict[str, str], 
    output_dir: str,
    cell_barcode_tag: str
) -> None:

    opened_output_bams = dict()
    with AFile(bam_file, "rb") as bam:
        for seg in bam:
            if seg.has_tag(cell_barcode_tag):
                barcode = seg.get_tag(cell_barcode_tag) 
                cmb = cmb_dict.get(barcode, '')
                if cmb:
                    if cmb not in opened_output_bams:
                        opened_output_bams[cmb] = AFile(
                            path.join(output_dir, f"{cmb}.bam"), 
                            "wb", template=bam
                        )
                    opened_output_bams[cmb].write(seg)
    for f in opened_output_bams.values():
        f.close()


def main(
    celltype_bams_dir: str, 
    output_dir: str, 
    cmb_csv_file: str, 
    cell_barcode_tag: str="CB"
) -> None:  
    
    bam_files = glob(path.join(celltype_bams_dir, "*.bam"))

    with open(cmb_csv_file, "rt") as f:
        cmb_dict = dict()
        for ln in f:
            barcode, cmb = ln.strip().split(',')
            cmb_dict[barcode] = cmb

    f = partial(
        split_bam, 
        cmb_dict=cmb_dict, 
        output_dir=output_dir, 
        cell_barcode_tag=cell_barcode_tag
    )     
    with Pool(min(os.cpu_count(), len(bam_files))) as pp:
        pp.map(f, bam_files)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
