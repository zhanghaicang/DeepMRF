
model_type='full_resn1d'
job_dir='full_v3'
rm -rf $job_dir
gpu_label=2
l2_reg=0.001
learn_rate=0.001
input_config=input.json
model_config=model_config.json
python main.py --model_type ${model_type} --mode train --op_alg adam  --epoch 1024 --learn_rate ${learn_rate} --l2_reg ${l2_reg} --model_config ${model_config}  --job_dir ${job_dir} --gpu_label ${gpu_label} --input_config ${input_config}
