docs=$(cat << EndOfDocs
${G}Usage:${RESET}
    bash $BASH_SOURCE <options> <raw BAM>

${G}Description:${RESET}
    This utility will be used to perform some essential preprocessing
    on the input <raw BAM> file, including:
        1. Extract primary alignment positions for each read while
           filtering out unwanted reads based on the provided lists
           of barcodes;
        2. Split BAM file by cell type;

${G}Options:${RESET}
    --output, -O <DIR>:
        Directory for saving the split BAM files.
    --barcode-celltype-mapping, -m <FILE>:
        CSV file containing cell barcodes and their corresponding cell types.
    --help, -h:
        Print this message, exit and return a non-zero exit status.

EndOfDocs
)
[[ $# -eq 0 || $1 == '-h' || $1 == '--help' ]] && USAGE


script_name=$(basename $BASH_SOURCE)
paras=$(getopt -o m:O: -l output:,barcode-celltype-mapping: \
        -n $script_name -- "$@") || USAGE
eval set -- $paras
while [[ $1 ]]
do
    case $1 in
        -O | --output)
            ARG_CK "$2" -O/--output
            output=$2
            INFO $script_name "output=$output"
            shift 2
            ;;
        --barcode-celltype-mapping | -m)
            check "$2" -m/--barcode-celltype-mapping
            bc_ct_csv=$2
            INFO $script_name "barcode-celltype-mapping=$bc_ct_csv"
            shift 2
            ;;
        --)
            shift
            break
            ;;
    esac
done
if [[ $1 ]]; then
    raw_bam_file=$1
    INFO $script_name "raw_bam=$raw_bam_file"
else
    ERROR $script_name "Missing required parameters"
fi
if [[ -z $output || -z $bc_ct_csv ]]; then
    ERROR $script_name "Missing required parameters"
fi


INFO $script_name "running ..." && 
    $PY $MAIN_DIR/src/py/filter_split_bam.py $raw_bam_file $output $bc_ct_csv &&
    INFO $script_name "finished"
