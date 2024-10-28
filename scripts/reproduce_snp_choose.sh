
source activate pytorch_v1
echo "change to conda in gpu server"
echo "----begin snp choose------"

exp_dir=exp_demo


python dg_rdg/mask_sublist_based_snp_choose.py --method mask_weight --conf $exp_dir/config_test.yml

