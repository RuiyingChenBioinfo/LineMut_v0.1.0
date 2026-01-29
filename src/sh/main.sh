#!/usr/bin/env bash


## --------------------- global objects ---------------------
set -a

SN='linemut_call'
R='\033[91m'
G='\033[92m'
Y='\033[93m'
RESET='\033[0m'
MAIN_DIR=$(dirname "$BASH_SOURCE")

function USAGE {
    echo -e "$_docs_" >&2
    exit 1
}

function ERROR {
    local message=$1
    echo -e "$R$(date +"%F %T") ERROR:$RESET [$_pn_] $message" >&2
    exit 1
}

function INFO {
    local message=$1
    echo -e "$G$(date +"%F %T") INFO:$RESET [$_pn_] $message"
}

function ARG_CK {
    [[ $1 =~ ^- ]] && ERROR "option $2 is missing its argument"
}
function SORT_VCF {
    local vcf_fp=$1
    if [[ $vcf_fp =~ ^.*_sorted\.vcf$ ]]; then return; fi
    local output_dir=$(dirname $vcf_fp)
    local fn=$(basename $vcf_fp)
    local output_vcf_fp=${output_dir}/${fn:0:-4}_sorted.vcf
    grep ^# $vcf_fp > $output_vcf_fp &&
        grep -v ^# $vcf_fp | sort -k1,1 -k2,2n >> $output_vcf_fp &&
        rm -f $vcf_fp
}

set +a


## ----------------------- help message -----------------------
_docs_=$(cat << EndOfDocs
${G}Usage:${RESET}
    ${SN} [OPTIONS]


${G}Description:${RESET}
    ${SN} is designed for detecting expressed single-nucleotide variants
    from single-cell RNA-sequencing or spatial transcriptomics data.


${G}Options:${RESET}
    --bam, -I:
        raw BAM file.
    --ref, -R:
        The reference genome sequence file in FASTA format.
    --output, -O:
        Directory for saving the result.
    --barcode-celltype-mapping, -m:
        The CSV file containing cell barcodes and their corresponding cell types.

    --cells-coordinate, -c:
        (optional) The CSV file containing cell coordinate information. If this
        parameter is not provided, the default behavior is to use cell type as
        the unit for mutation detection without CMB partitioning.
    --known-variants-dir, -v:
        (optional) A directory of VCF-formatted known variant sites for the species.
    --k-mer, -k:
        (optional) The length of k-mer, default: 9.
    --cell-barcode-tag, -t:
        (optional) The tag name denoting the cell barcode in the BAM file
        defaults to 'CB'.
    --python:
        (optional) The pathname to the Python interpreter you want to use.
        By default, it uses the first 'python3' found in the PATH.
    --gatk:
        (optional) The pathname of the GATK executable file. By default,
        search for 'gatk' in the PATH.
    --samtools:
        (optional) The pathname of the samtools. By default, search for the
        'samtools' in the PATH.

    --help, -h:
        Print this message, exit and return a non-zero exit status.

EndOfDocs
)
[[ $# -eq 0 || $1 == '-h' || $1 == '--help' ]] && USAGE


## ----------------------- argument parsing -----------------------
_pn_=$(basename $BASH_SOURCE)
paras=$(getopt -o I:m:O:c:k:t:R:v: \
-l output:,barcode-celltype-mapping:,bam:,python:,gatk:,samtools:,\
cells-coordinate:,k-mer:,cell-barcode-tag:,ref:,known-variants-dir: \
-n $SN -- "$@") || USAGE
eval set -- $paras
k=9
cb_tag='CB'
while [[ $1 ]]
do
    case $1 in
        -O | --output)
            ARG_CK "$2" "-O/--output"
            output=$2
            INFO "output=$output"
            shift 2
            ;;
        --barcode-celltype-mapping | -m)
            ARG_CK "$2" "-m/--barcode-celltype-mapping"
            bc_ct_csv=$2
            INFO "barcode-celltype-mapping=$bc_ct_csv"
            shift 2
            ;;
        --bam | -I)
            ARG_CK "$2" "-I/--bam"
            raw_bam=$2
            INFO "bam=$raw_bam"
            shift 2
            ;;
        --python)
            ARG_CK "$2" "--python"
            export PY=$2
            INFO "python=$PY"
            shift 2
            ;;
        --gatk)
            ARG_CK "$2" "--gatk"
            export GATK=$2
            INFO "gatk=$GATK"
            shift 2
            ;;
        --samtools)
            ARG_CK "$2" "--samtools"
            export SAMT=$2
            INFO "samtools=$SAMT"
            shift 2
            ;;
        --cells-coordinate | -c)
            ARG_CK "$2" "-c/--cells-coordinate"
            cells_coor=$2
            INFO "cells-coordinate=$cells_coor"
            shift 2
            ;;
        --k-mer | -k)
            ARG_CK "$2" "-k/--k-mer"
            k=$2
            INFO "k-mer=$k"
            shift 2
            ;;
        --cell-barcode-tag | -t)
            ARG_CK "$2" "-t/--cell-barcode-tag"
            cb_tag=$2
            INFO "cell-barcode-tag=$cb_tag"
            shift 2
            ;;
        --ref | -R)
            ARG_CK "$2" "-R/--ref"
            ref=$2
            INFO "ref=$ref"
            shift 2
            ;;
        --known-variants-dir | -v)
            ARG_CK "$2" "-v/--known-variants-dir"
            kn_vars_dir=$2
            INFO "known-variants-dir=$kn_vars_dir"
            shift 2
            ;;
        --)
            shift
            break
            ;;
    esac
done
if [[ -z $PY ]]; then
    msg="no available 'python3' interpreter found in the PATH"
    PY=$(command -v python3) && export PY && INFO "python=$PY" || ERROR "$msg"
fi

if [[ -z $GATK ]]; then
    msg="no available 'gatk' found in the PATH"
    GATK=$(command -v gatk) && export GATK && INFO "gatk=$GATK" || ERROR "$msg"
fi

if [[ -z $SAMT ]]; then
    msg="No available 'samtools' found in the PATH"
    SAMT=$(command -v samtools) && export SAMT &&
        INFO "samtools=$SAMT" || ERROR "$msg"
fi
if [[ -z $cells_coor ]]; then
    celltype_level="yes"
    INFO "celltype_level=yes"
fi
if [[ -d $output ]]; then
    ERROR "The output directory '$output' already exists."
else
    mkdir $output
fi

## ----------------------- workflow -----------------------
set -e
declare -i step_no=0

step_no+=1
INFO "Step $step_no: bam filtering and cell type-based demultiplexing"
INFO "Step $step_no: running ..."
celltype_bams_dir="$output/${step_no}.celltype_bams"
mkdir $celltype_bams_dir
$PY $MAIN_DIR/src/py/filter_split_bam.py \
    $raw_bam $celltype_bams_dir $bc_ct_csv $cb_tag
ls $celltype_bams_dir/*.bam | xargs -n1 -P$(nproc) $SAMT index
split_bams_dir=$celltype_bams_dir
bc_unit_map_file=$bc_ct_csv
INFO "Step $step_no: finished"

if [[ $celltype_level != "yes" ]]; then
    step_no+=1
    INFO "Step $step_no: divide CMBs and split the bam files accordingly."
    INFO "Step $step_no: running ..."
    cmb_result_dir="$output/${step_no}.cellmetabin_result"
    mkdir $cmb_result_dir
    bash $MAIN_DIR/src/sh/divide_cmb.sh \
        $celltype_bams_dir $k $cmb_result_dir $cells_coor $cb_tag
    ls $cmb_result_dir/cmb_bams/*.bam | xargs -n1 -P$(nproc) $SAMT index
    split_bams_dir=${cmb_result_dir}/cmb_bams
    bc_unit_map_file=${cmb_result_dir}/${k}mer_cmbs/cellmetabin.csv
    INFO "Step $step_no: finished"
fi

step_no+=1
INFO "Step $step_no: Running variant calling in parallel using freebayes."
INFO "Step $step_no: running ..."
freebayes_vcfs_dir=${output}/${step_no}.freebayes_vcfs
mkdir $freebayes_vcfs_dir
bash $MAIN_DIR/src/sh/run_freebayes.sh $ref $split_bams_dir $freebayes_vcfs_dir
INFO "Step $step_no: finished"

step_no+=1
INFO "Step $step_no: Running variant calling in parallel using Strelka2."
INFO "Step $step_no: running ..."
strelka_vcfs_dir=${output}/${step_no}.strelka_vcfs
mkdir $strelka_vcfs_dir
bash $MAIN_DIR/src/sh/run_strelka.sh $ref $split_bams_dir $strelka_vcfs_dir
INFO "Step $step_no: finished"

if [[ -z $kn_vars_dir ]]; then
    step_no+=1
    INFO "Step $step_no: Integrate Strelka and Freebayes results."
    INFO "Step $step_no: running ..."
    integrated_vars_dir=${output}/${step_no}.integrated_vars
    mkdir $integrated_vars_dir
    $PY $MAIN_DIR/src/py/intersect_vcf.py \
        $integrated_vars_dir $freebayes_vcfs_dir $strelka_vcfs_dir
    find $integrated_vars_dir -name "*.vcf" -type f | \
        xargs -I% -P$(nproc) bash -c "SORT_VCF %"
    kn_vars_dir=${integrated_vars_dir}/trusted
    flag='no_kn'
    INFO "Step $step_no: finished"
else
    flag='kn'
fi

step_no+=1
INFO "Step $step_no: Running variant calling in parallel using GATK."
INFO "Step $step_no: running ..."
gatk_result_dir=$output/${step_no}.gatk_result
mkdir $gatk_result_dir
bash $MAIN_DIR/src/sh/run_gatk.sh \
    $ref $split_bams_dir $gatk_result_dir $kn_vars_dir $flag
INFO "Step $step_no: finished"

step_no+=1
INFO "Step $step_no: Categorize variants by confidence level."
INFO "Step $step_no: running ..."
categorized_result_dir=$output/${step_no}.categorized_vars
mkdir $categorized_result_dir
$PY $MAIN_DIR/src/py/intersect_vcf.py \
    $categorized_result_dir \
    $freebayes_vcfs_dir \
    $strelka_vcfs_dir \
    $gatk_result_dir/filtered_vcfs
find $categorized_result_dir -name "*.vcf" -type f | \
    xargs -I% -P$(nproc) bash -c "SORT_VCF %"
INFO "Step $step_no: finished"

step_no+=1
INFO "Step $step_no: Filter variant sites and produce the final h5ad file."
INFO "Step $step_no: running ..."
h5ad_files_dir=$output/${step_no}.h5ad_files
mkdir $h5ad_files_dir
$PY $MAIN_DIR/src/py/generate_final_h5ad.py \
    $categorized_result_dir \
    $split_bams_dir \
    $h5ad_files_dir \
    $cb_tag \
    $bc_unit_map_file
INFO "Step $step_no: finished"
