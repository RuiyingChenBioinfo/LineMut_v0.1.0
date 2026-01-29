set -e

celltype_bams_dir=$1
k=$2
result_dir=$3
cells_coor_csv=$4
cb_tag=$5
_pn_=$(basename $BASH_SOURCE)


INFO "divide CMBs ..."
cmb_output_dir=${result_dir}/${k}mer_cmbs
mkdir $cmb_output_dir
$PY $MAIN_DIR/src/py/cellmetabin.py \
    $celltype_bams_dir $k $cmb_output_dir $cells_coor_csv $cb_tag
INFO "divide CMBs finished"

INFO "split BAMs ..."
output_dir=${result_dir}/cmb_bams
mkdir $output_dir
$PY $MAIN_DIR/src/py/split_bam_by_cmb.py \
    $celltype_bams_dir $output_dir ${cmb_output_dir}/cellmetabin.csv $cb_tag
INFO "split BAMs finished"

