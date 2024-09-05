#python3 featureVisulize.py --inlier_features_path "/features/cifar10_resnet18_temp_0.01_id_0_lr_0.001_bz_512_train" --outlier_features_path "/features/cifar10_resnet18_temp_0.01_id_0_lr_0.001_bz_512_test_unknown" --num_classes 6 --save_path "/plots/cifar10_resnet18_temp_0.01_id_0_lr_0.001_bz_512_unknown_tsne.pdf"

#python3 featureVisulize.py --inlier_features_path "/features/cifar10_resnet18_temp_0.05_id_5_lr_0.001_bz_256_train" --num_classes 10 --save_path "/plots/cifar10_resnet18_temp_0.05_id_5_lr_0.001_bz_256_train_tsne.pdf" --reduced_len 0


python3 featureVisulize.py --inlier_features_path "/features/cifar10_resnet18_temp_0.05_id_5_lr_0.01_bz_256_train" --num_classes 10 --save_path "/plots/cifar10_resnet18_temp_0.05_id_5_lr_0.01_bz_256_train_tsne.pdf" --reduced_len 0

#python3 featureVisulize.py --inlier_features_path "/features/svhn_resnet18_temp_0.05_id_0_lr_0.001_bz_256_train" --num_classes 6 --save_path "/plots/svhn_resnet18_temp_0.05_id_0_lr_0.001_bz_256_train_tsne.pdf" --reduced_len 0

#python3 featureVisulize.py --inlier_features_path "/features/tinyimgnet_resnet18_temp_0.05_id_0_lr_0.001_bz_512_train" --num_classes 20 --save_path "/plots/tinyimgnet_resnet18_temp_0.05_id_0_lr_0.001_bz_512_train_tsne.pdf" --reduced_len 0

