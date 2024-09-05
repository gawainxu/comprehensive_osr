#for i in {0..31}
#do
#   echo $i
#   python3 gradcam.py --data_id $i --class_idx 5 --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_1_original_data__mixup_positive_alpha_10_beta_0.3_cutmix_no_SimCLR_0.0_1.0_0.05_trail_0_128_256/last_linear.pth"
#done


#for i in {0..127}
#do
#   echo $i
#   python3 gradcam.py --data_id 91 --feature_id $i --class_idx 5 --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_1_original_data__mixup_positive_alpha_10_beta_0.3_cutmix_no_SimCLR_1.0_1.0_0.05_trail_0_128_256/last_linear.pth"
#done



for i in {0..127}
do
   echo $i
   python3 gradcam.py --mode "sim" --data_id $i --class_idx 5 --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_1_original_data__mixup_positive_alpha_10_beta_0.3_cutmix_no_SimCLR_1.0_1.0_0.05_trail_0_128_256/last.pth"
done
