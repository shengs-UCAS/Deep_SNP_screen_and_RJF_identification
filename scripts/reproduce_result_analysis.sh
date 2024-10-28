
source activate torch_gpu_3
echo "change to conda in gpu server"
echo "----begin to analysis result------"

exp_dir=exp_demo_1

model_dir=./$exp_dir/model_dir
logger_file=$model_dir/analysis.log
result_file=$model_dir/pred_all

python -u dg_rdg/pred_result_ana.py --method confusion_matrix_detail  --logger_file $logger_file --result_file $result_file 
