set -a 
REF=$1
OUT_DIR=$3
STRELKA=$(realpath ${MAIN_DIR}/bin/strelka_germline)
_pn_=$(basename $BASH_SOURCE)

function strelka {
    local bam_file=$1
    local fn=$(basename $bam_file)
    local output_dir=${OUT_DIR}/${fn:0:-4}
    mkdir $output_dir
    $STRELKA --bam $bam_file --reference $REF --runDir $output_dir \
        > $output_dir/log.txt 2>&1
    $output_dir/runWorkflow.py -m local -j 8 >> $output_dir/log.txt 2>&1
}

set +a

split_bams_dir=$2
ls ${split_bams_dir}/*.bam | xargs -I% -P$(nproc) bash -c "strelka %"
