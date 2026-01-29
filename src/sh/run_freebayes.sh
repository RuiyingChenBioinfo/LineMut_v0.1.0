set -a

REF=$1
OUT_DIR=$3
FREEBAYES=$(realpath ${MAIN_DIR}/bin/freebayes)
_pn_=$(basename $BASH_SOURCE)

function freebayes {
    local bam_file=$1
    local fn=$(basename $bam_file)
    $FREEBAYES -f $REF -C 5 -F 0.1 $bam_file > ${OUT_DIR}/${fn:0:-4}.vcf
}

set +a

split_bams_dir=$2
ls ${split_bams_dir}/*.bam | xargs -I% -P$(nproc) bash -c "freebayes %"
