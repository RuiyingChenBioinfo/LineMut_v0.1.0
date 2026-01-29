import os
import sys
from os import path
from glob import iglob
from multiprocessing import Pool

import vcf


class SNV:
    __slots__ = "chrom", "pos", "ref", "alt"
    def __init__(self, chrom, pos, ref, alt):
        self.chrom = chrom
        self.pos = int(pos)
        self.ref = ref.upper()
        self.alt = alt.upper()

    def __hash__(self):
        return self.pos

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.chrom == other.chrom and
                    self.pos == other.pos and 
                    self.alt == other.alt)
        else:
            raise TypeError(
                "'==' not supported between instances of "
                f"'{type(self)}' and '{type(other)}'"
            )

    def __repr__(self):
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"


def get_ref_snvs(vcf_lst: tuple[str]) -> tuple[set]:
    snvs_lst = list()
    for f in vcf_lst:
        reader = vcf.Reader(filename=f)
        snvs = set()
        for rcd in reader:
            for alt in rcd.ALT:
                snvs.add(SNV(rcd.CHROM, rcd.POS, rcd.REF, alt.sequence)) 
        snvs_lst.append(snvs)
    if len(vcf_lst) == 2:
        return snvs_lst[1], snvs_lst[0]
    elif len(vcf_lst) == 3:
        return (snvs_lst[1] | snvs_lst[2], 
                snvs_lst[0] | snvs_lst[2], 
                snvs_lst[0] | snvs_lst[1])
    else:
        raise ValueError(
            "supporting more than three VCF files is not available."
        )
 

def proc_grouped_vcfs(label: str, output_dir: str, *vcfs: tuple[str]) -> None:
    os.makedirs(trust_out := path.join(output_dir, f"trusted/{label}"))
    os.makedirs(untrust_out := path.join(output_dir, f"untrusted/{label}"))
    ref_snvs = get_ref_snvs(vcfs)
    already_output_snvs = set()
    for i, (f, snvs) in enumerate(zip(vcfs, ref_snvs)):
        reader = vcf.Reader(filename=f)
        if 0 == i:
            trusted_snvs_vcf = vcf.Writer(
                open(path.join(trust_out, f"{label}_trusted.vcf"), "wt"), 
                template=reader
            )
            untrusted_snvs_vcf = vcf.Writer(
                open(path.join(untrust_out, f"{label}_untrusted.vcf"), "wt"), 
                template=reader
            )
        for rcd in reader:
            if rcd.is_snp:
                alt_snvs = [
                    SNV(rcd.CHROM, rcd.POS, rcd.REF, alt.sequence) 
                    for alt in rcd.ALT
                ]
                if all(s in snvs for s in alt_snvs):
                    if not all(s in already_output_snvs for s in alt_snvs):
                        trusted_snvs_vcf.write_record(rcd)
                        already_output_snvs.update(alt_snvs)
                else:
                    untrusted_snvs_vcf.write_record(rcd)
    trusted_snvs_vcf.close()
    untrusted_snvs_vcf.close()
    

def main(
    output_dir: str, 
    freebayes_dir: str, 
    strelka_dir: str, 
    gatk_dir: str=''
) -> None:

    os.makedirs(path.join(output_dir, "trusted"))
    os.makedirs(path.join(output_dir, "untrusted"))

    labels = set(
        path.basename(fp).replace(".vcf", '') 
        for fp in iglob(path.join(freebayes_dir, "*.vcf"))
    )
    assert labels == set(os.listdir(strelka_dir))
    if gatk_dir:
        assert labels == set(
            path.basename(fp).replace(".vcf", '')
            for fp in iglob(path.join(gatk_dir, "*.vcf"))
        )

    def args():
        for label in labels:
            if gatk_dir:
                args = (
                    label, output_dir, path.join(freebayes_dir, f"{label}.vcf"), 
                    path.join(strelka_dir, f"{label}/results/variants/variants.vcf.gz"), 
                    path.join(gatk_dir, f"{label}.vcf")
                )
            else:
                args = (
                    label, output_dir, path.join(freebayes_dir, f"{label}.vcf"), 
                    path.join(strelka_dir, f"{label}/results/variants/variants.vcf.gz"), 
                )
            yield args

    with Pool(min(os.cpu_count(), len(labels))) as pp:
        pp.starmap(proc_grouped_vcfs, args())
        

if __name__ == "__main__":
    main(*sys.argv[1:])
    # proc_grouped_vcfs(*sys.argv[1:])
