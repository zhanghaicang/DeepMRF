

model_type='full_resn_mrf'
job_dir='/data/hz2529/data/missense/protein/res/mrf_v14'
mkdir -p $job_dir
gpu_label=1
l2_reg=0.001
learn_rate=0.001
mrf_1d_reg=0.0
mrf_2d_reg=0.00
batch_size=32
for i in `seq 9 9`
do
    python main.py --model_type ${model_type} --mode train --op_alg adam  --epoch 1024 --batch_size ${batch_size} --learn_rate ${learn_rate} --l2_reg ${l2_reg} --model_config ./model_config.json  --job_dir ${job_dir}/cv${i} --gpu_label ${gpu_label}  --input_config ./input_cv2.json  --cv $i --mrf_1d_reg ${mrf_1d_reg} --mrf_2d_reg ${mrf_2d_reg} > $job_dir/cv_${i}.log 
done
