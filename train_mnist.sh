#python3 main_supcon.py --epochs 20 --save_freq 1 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_epochs "1000" --lr_decay_rate 0.8 --model "resnet18" --datasets "mnist" --size 32 --trail 0 --temp 0.05 --method "SimCLR" --method_lam 1.2

#python3 main_supcon.py --epochs 10 --save_freq 1 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_epochs "1000" --lr_decay_rate 0.8 --model "resnet18" --datasets "mnist" --size 32 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0


python3 main_supcon.py --epochs 10 --save_freq 1 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_epochs "1000" --lr_decay_rate 0.8 --model "resnet18" --datasets "mnist" --size 32 --trail 0 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 0.0 --randaug 0 --mixup_positive False
