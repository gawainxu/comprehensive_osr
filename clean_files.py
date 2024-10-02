import os

root_path = "/home/
dataset_folders = os.listdir(root_path)
print(dataset_folders)

for dataset_folder in dataset_folders:
    curr_path = os.path.join(root_path, dataset_folder)
    model_folders = os.listdir(curr_path)
    for model_folder in model_folders:
        curr_path = os.path.join(root_path, dataset_folder, model_folder)
        print(curr_path)
        files = os.listdir(curr_path)
        os.chdir(curr_path)
        for file in files:
            if "last" not in file or "linear" not in file or "loss" not in file:
                os.remove(file)
