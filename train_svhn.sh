# 2749
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 0 --temp 0.05
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 1 --temp 0.05
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 2 --temp 0.05
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 3 --temp 0.05
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 4 --temp 0.05

#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 0 --temp 0.01
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 1 --temp 0.01
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 2 --temp 0.01
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 3 --temp 0.01
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 4 --temp 0.01



python3 main_supcon.py --epochs 300 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 256 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 0 --method_lam 1.0 --randaug 0 --mixup_positive True --positive_method "layersaliencymix" --augmentation_method "mixup_positive" --alpha_vanilla 1.0 --beta_vanilla 1.0


#python3 main_supcon.py --epochs 300 --batch_size 512 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 0 --temp 0.05


#python3 main_supcon_flexible_tau.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 0 --tau_strategy "fixed_set"
#python3 main_supcon_flexible_tau.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 1 --tau_strategy "fixed_set"
#python3 main_supcon_flexible_tau.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 2 --tau_strategy "fixed_set"
#python3 main_supcon_flexible_tau.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 3 --tau_strategy "fixed_set"
#python3 main_supcon_flexible_tau.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --datasets "svhn" --size 32 --trail 4 --tau_strategy "fixed_set"


