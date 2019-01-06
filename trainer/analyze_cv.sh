
for i in `seq 0 9`; do python select_cv_model.py --train_log test_cv/cv_${i}.log; done | cut -f 2 -d '=' | cut -f 2 -d ' '  > cv_models.txt

bash test_cv.sh

python calc_cv_acc.py --base_dir cv_res_v2 --input_list casp12.list
python calc_cv_acc.py --base_dir cv_res_v2 --input_list cameo.list
