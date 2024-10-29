
source activate pytorch_v1
echo "change to conda in gpu server"
echo "----begin to analysis result------"

exp_dir=trained_model


# prepare-data.pickle

echo "----begin identify model pred------"
mkdir trained_model_pred

read -r -p "continue?" input

vcf_file=demo_data/demo.vcf
used_snp_file=trained_model/final_model_700snp.snp_tokens

echo "----VCF file SNP Check------"
python dg_rdg/rjf_sample_feat_parse.py --method stat_vcf  --vcf_file $vcf_file --snp_file $used_snp_file  

read -r -p "continue?" input
echo "---VCF to AI Dataset---"
data_file=trained_model_pred/new_sample.pickle
python dg_rdg/rjf_sample_feat_parse.py --method parse_vcf --vcf_file $vcf_file --data_file $data_file --mock_group 1

read -r -p "continue?" input
echo "---do pred---"

checkpoint=trained_model/snp_choose_main.model_checkpoint
pred_log=trained_model_pred/new_sample_pred.log
pred_result=trained_model_pred/pred_result

python dg_rdg/rdc_classify_test.py --checkpoint $checkpoint --logger_file $pred_log  --data $data_file --test_result_file $pred_result

