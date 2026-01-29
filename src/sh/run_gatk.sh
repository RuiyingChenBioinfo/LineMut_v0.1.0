set -a 

REF=$1
OUT_DIR=$3
KN_VAR_DIR=$4
FLAG=$5
LOG_DIR=${OUT_DIR}/logs
SPLIT_RESULT_DIR=${OUT_DIR}/split_ncigar_bams
BQSR_RESULT_DIR=${OUT_DIR}/bqsr
RAW_VCFS=${OUT_DIR}/raw_vcfs
FILTERED_VCFS=${OUT_DIR}/filtered_vcfs
_pn_=$(basename $BASH_SOURCE)

function gatk {
    local bam_file=$1
    local fn=$(basename $bam_file)
    local bare_fn=${fn:0:-4}
    local split_bam=${SPLIT_RESULT_DIR}/$fn
    local log_dir=${LOG_DIR}/${bare_fn}
    mkdir $log_dir

    $GATK SplitNCigarReads -R $REF -I $bam_file -O $split_bam \
        > ${log_dir}/split_ncigar.log 2>&1

    if [[ $FLAG == 'no_kn' ]]; then
        kn_v_d=${KN_VAR_DIR}/${bare_fn}
    else
        kn_v_d=${KN_VAR_DIR}
    fi

    for vcf in $(ls ${kn_v_d}/*.vcf); do 
        known_sites+=" --known-sites $vcf"
    done
    local recal_table="${BQSR_RESULT_DIR}/${bare_fn}_recal_table"
    local bqsr_bam=${BQSR_RESULT_DIR}/${bare_fn}_bqsr.bam
    $GATK BaseRecalibrator -I $split_bam -R $REF -O $recal_table \
        $known_sites > ${log_dir}/base_recalibrator.log 2>&1
    $GATK ApplyBQSR --add-output-sam-program-record -R $REF \
        -I $split_bam --use-original-qualities -O $bqsr_bam \
        --bqsr-recal-file $recal_table > "${log_dir}/apply_bqsr.log" 2>&1 

    local raw_vcf=${RAW_VCFS}/${bare_fn}.vcf
    $GATK HaplotypeCaller -R $REF -I $bqsr_bam -O $raw_vcf \
        --dont-use-soft-clipped-bases \
        --standard-min-confidence-threshold-for-calling 20 \
        > "${log_dir}/haplotypecaller.log" 2>&1
    $GATK VariantFiltration --R "$REF" --V $raw_vcf --filter-name "FS" \
        --filter "FS > 30.0" --filter-name "QD" --filter "QD < 2.0" \
        -O "${FILTERED_VCFS}/${bare_fn}.vcf" > ${log_dir}/filter_vcf.log 2>&1
}

set +a 

mkdir $LOG_DIR $BQSR_RESULT_DIR $SPLIT_RESULT_DIR $RAW_VCFS $FILTERED_VCFS

$SAMTOOLS faidx $REF > ${LOG_DIR}/faidx.log 2>&1
$GATK CreateSequenceDictionary -R $REF \
    > "${LOG_DIR}/create_fasta_dictionary.log" 2>&1
# ls ${KN_VAR_DIR}/**/*.vcf | xargs -P5 -I% $GATK IndexFeatureFile -I "%" \
#     > "${LOG_DIR}/index_known_variants.log" 2>&1
find ${KN_VAR_DIR} -name "*.vcf" -type f | xargs -P5 -I% $GATK IndexFeatureFile -I "%" \
    > "${LOG_DIR}/index_known_variants.log" 2>&1

split_bams_dir=$2 
ls ${split_bams_dir}/*.bam | xargs -I% -P$(nproc) bash -c "gatk %"
