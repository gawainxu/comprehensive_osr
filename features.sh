#python3 feature_reading.py --dataset "cifar-10-100-10" --model "resnet18" --model_path "/save/SupCon/cifar-10-100-10_models/cifar-10-100-10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_0.0_1.0_0.05_trail_3_128_256/ckpt_epoch_0.pth" --epoch 0 --trail 2 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train" 
#python3 feature_reading.py --dataset "cifar-10-100-10" --model "resnet18" --model_path "/save/SupCon/cifar-10-100-10_models/cifar-10-100-10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_2_128_256_twostage/ckpt_epoch_0.pth" --epoch 0 --trail 2 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known" 
#python3 feature_reading.py --dataset "cifar-10-100-50" --model "resnet18" --model_path "/save/SupCon/cifar-10-100-10_models/cifar-10-100-10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_2_128_256_twostage/ckpt_epoch_0.pth" --epoch 0 --trail 2 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown" 



#python3 feature_reading.py --dataset "cifar-10-100-10" --model "resnet18" --model_path "/save/SupCon/cifar-10-100-10_models/cifar-10-100-10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_2_128_256_twostage/ckpt_epoch_0.pth" --epoch 0 --trail 2 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown" 



#python3 feature_reading.py --dataset "cifar-10-100-10" --model "resnet18" --model_path "/save/SupCon/cifar-10-100-10_models/cifar-10-100-10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_0_128_256/ckpt_epoch_500.pth" --epoch 500 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown" 



#python3 feature_reading.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_2_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_2_128_256/ckpt_epoch_800.pth" --epoch 800 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train" 
#python3 feature_reading.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_2_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_2_128_256/ckpt_epoch_800.pth" --epoch 800 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known" 
#python3 feature_reading.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_2_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_2_128_256/ckpt_epoch_800.pth" --epoch 800 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown" 



#python3 feature_reading.py --dataset "cifar100" --model "resnet18" --model_path "/save/SupCon/cifar100_models/cifar100_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_0_128_256_twostage/last.pth" --epoch 400 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train" 
#python3 feature_reading.py --dataset "cifar100" --model "resnet18" --model_path "/save/SupCon/cifar100_models/cifar100_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_0_128_256_twostage/last.pth" --epoch 400 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known"



#python3 feature_reading.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_6_128_256/ckpt_epoch_500.pth" --epoch 500 --trail 6 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train" 
#python3 feature_reading.py --dataset "tinyimgnet" --model "resnet18" --model_path "/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_6_128_256/ckpt_epoch_500.pth" --epoch 500 --trail 6 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known"    


#python3 feature_reading.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_5_128_256/ckpt_epoch_450.pth" --epoch 450 --trail 5 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train" 
#python3 feature_reading.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_5_128_256/ckpt_epoch_450.pth" --epoch 450 --trail 5 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known"


#python3 feature_reading.py --dataset "svhn" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_5_128_256/ckpt_epoch_100.pth" --epoch 100 --trail 5 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known"           
#python3 feature_reading.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_5_128_256/ckpt_epoch_500.pth" --epoch 500 --trail 5 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known" 

#python3 feature_reading.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_5_128_256/ckpt_epoch_400.pth" --epoch 400 --trail 5 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train" 
#python3 feature_reading.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_5_128_256/ckpt_epoch_400.pth" --epoch 400 --trail 5 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known" 
 

#python3 feature_reading.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_5_128_256/ckpt_epoch_500.pth" --epoch 500 --trail 5 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known" 
#python3 feature_reading.py --dataset "cifar10" --model "resnet18" --model_path "/save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.2_0.05_trail_5_128_256/ckpt_epoch_560.pth" --epoch 560 --trail 5 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known"


#python3 feature_reading.py --dataset "mnist" --model "resnet18" --model_path "/save/SupCon/mnist_models/mnist_resnet18_original_data__vanilia__SimCLR_1.0_0.8_0.05_trail_0_128_256/ckpt_epoch_4.pth" --epoch 4 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train" 
#python3 feature_reading.py --dataset "mnist" --model "resnet18" --model_path "/save/SupCon/mnist_models/mnist_resnet18_original_data__vanilia__SimCLR_1.0_0.8_0.05_trail_0_128_256/ckpt_epoch_4.pth" --epoch 4 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known" 
#python3 feature_reading.py --dataset "mnist" --model "resnet18" --model_path "/save/SupCon/mnist_models/mnist_resnet18_original_data__vanilia__SimCLR_1.0_0.8_0.05_trail_0_128_256/ckpt_epoch_4.pth" --epoch 4 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown" 


#python3 feature_reading.py --dataset "mnist" --model "resnet18" --model_path "/save/SupCon/mnist_models/mnist_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_0_128_256/ckpt_epoch_20.pth" --epoch 20 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train" 
#python3 feature_reading.py --dataset "mnist" --model "resnet18" --model_path "/save/SupCon/mnist_models/mnist_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_0_128_256/ckpt_epoch_20.pth" --epoch 20 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known" 
#python3 feature_reading.py --dataset "mnist" --model "resnet18" --model_path "/save/SupCon/mnist_models/mnist_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_0_128_256/ckpt_epoch_20.pth" --epoch 20 --trail 0 --augmentation_method "mixup" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown" 



python3 feature_reading.py --dataset "mnist" --model "resnet18" --model_path "/save/SupCon/mnist_models/mnist_resnet18_original_data__vanilia__SimCLR_1.0_0.0_0.05_trail_0_128_256/ckpt_epoch_4.pth" --epoch 200 --trail 0 --augmentation_method "vanilia" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "train" 
python3 feature_reading.py --dataset "mnist" --model "resnet18" --model_path "/save/SupCon/mnist_models/mnist_resnet18_original_data__vanilia__SimCLR_1.0_0.0_0.05_trail_0_128_256/ckpt_epoch_4.pth" --epoch 200 --trail 0 --augmentation_method "vanilia" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_known" 
python3 feature_reading.py --dataset "mnist" --model "resnet18" --model_path "/save/SupCon/mnist_models/mnist_resnet18_original_data__vanilia__SimCLR_1.0_0.0_0.05_trail_0_128_256/ckpt_epoch_4.pth" --epoch 200 --trail 0 --augmentation_method "vanilia" --temp 0.05 --lr 0.01 --training_bz 256 --if_train "test_unknown" 

