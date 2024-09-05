#python3 main_supcon.py --epochs 500 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_epochs "2500" --model "resnet18" --datasets "cifar10" --size 32 --trail 5 --temp 0.05 --method "SimCLR" --method_lam 1.0 --mixup_vanilla True --augmentation_method "mixup_vanilla" --vanilla_method "random" --alpha_vanilla 0.1 --beta_vanilla 0.1
#python3 main_supcon.py --epochs 500 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_epochs "2500" --model "resnet18" --datasets "cifar10" --size 32 --trail 1 --temp 0.05 --method "SimCLR" --method_lam 3.0

#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_lam 1.0 --randaug True --argmentation_n 2 --argmentation_m 6 --apool True
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_lam 1.0 --randaug True --argmentation_n 1 --argmentation_m 8 --apool True
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_lam 1.0 --randaug False --apool True

#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug False
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet34" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug False


#python3 main_supcon.py --epochs 500 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0
#python3 main_supcon.py --epochs 500 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 2.2 --randaug 0
#python3 main_supcon.py --epochs 500 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 1 --argmentation_n 1 --argmentation_m 6
#python3 main_supcon.py --epochs 500 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 1 --argmentation_n 2 --argmentation_m 6
#python3 main_supcon.py --epochs 500 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 1 --argmentation_n 2 --argmentation_m 8


#python3 main_supcon.py --epochs 10 --save_freq 1 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar-10-100-10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --last_model_path "./save/SupCon/cifar-10-100-10_models/cifar-10-100-10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_0.0_1.0_0.05_trail_1_128_256/last.pth"



#python3 main_supcon.py --epochs 1 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar-10-100-10" --size 256 --trail 4 --temp 0.05 --method "SimCLR" --method_gama 0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "cutmix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --mixed_precision True


#python3 main_supcon.py --epochs 650 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar-10-100-10" --size 256 --trail 4 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 --last_model_path "./save/SupCon/cifar-10-100-10_models/cifar-10-100-10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_0.0_1.0_0.05_trail_4_128_256/ckpt_epoch_150.pth"


#python3 main_supcon.py --epochs 600 --save_freq 10 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 2 --temp 0.05 --method "SimCLR" --method_gama 0.0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0 

python3 main_supcon.py --epochs 10 --save_freq 2 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "cifar10" --size 256 --trail 1 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.2 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0  --last_model_path "./save/SupCon/cifar10_models/cifar10_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_0.0_1.0_0.05_trail_1_128_256_[2,3]/ckpt_epoch_600.pth"
