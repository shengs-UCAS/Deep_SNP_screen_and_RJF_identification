

#!/bin/zsh 
source activate pytorch_v1
echo "----begin prepare train file------"

exp_dir=exp_demo
vcf_file=demo_data/demo.vcf
sample_2_group_file=demo_data/demo.breeds_info
data_file=./$exp_dir/GGS_VS_DC_demo.pickle


python dg_rdg/rjf_sample_feat_parse.py --method parse_vcf --vcf_file $vcf_file --sample_2_group_file $sample_2_group_file --data_file $data_file


echo "----stat all data------"

python dg_rdg/rjf_sample_feat_parse.py --method stat --data_file $data_file --output_file $data_file


echo "----split train test------"

python dg_rdg/rjf_sample_feat_parse.py --method split_train_test --data_file $data_file