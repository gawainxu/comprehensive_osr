# 31443
#python3 main_supcon.py --epochs 300 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --dataset "tinyimgnet" --size 32 --trail 0 --temp 0.05

#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --dataset "tinyimgnet" --size 64 --trail 4 --temp 0.05


#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --dataset "tinyimgnet" --size 64 --trail 0 --temp 0.01
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --dataset "tinyimgnet" --size 64 --trail 1 --temp 0.01
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --dataset "tinyimgnet" --size 64 --trail 2 --temp 0.01
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --dataset "tinyimgnet" --size 64 --trail 3 --temp 0.01
#python3 main_supcon.py --epochs 400 --batch_size 256 --learning_rate 0.001 --lr_decay_epochs "25, 50, 75, 100, 125, 150, 175, 200" --lr_decay_rate 0.8 --model "resnet18" --dataset "tinyimgnet" --size 64 --trail 4 --temp 0.01

python3 main_supcon.py --epochs 800 --save_freq 50 --batch_size 256 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet18" --datasets "tinyimgnet" --size 256 --trail 5 --temp 0.05 --method "SimCLR" --method_gama 1.0 --method_lam 1.0 --randaug 0 --mixup_positive False
