import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class YOLODATASET(Dataset):
    def __init__(self, image_folder, label_folder, transform, label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.image_names = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        # 获得标注文件夹的路径
        label_name = image_name.split(".")[0] + ".txt"
        label_path = os.path.join(self.label_folder, label_name)
        with open(label_path, "r", encoding="utf-8") as f:
            label_context = f.read()
        object_infos = label_context.strip().split("\n")
        target = [0 for _ in range(7)]
        # 每个物体都取的版本
        # for object_info in object_infos:
        #     info_list = object_info.strip().split(" ")
        #     class_id = float(info_list[0])
        #     center_x = float(info_list[1])
        #     center_y = float(info_list[2])
        #     width = float(info_list[3])
        #     height = float(info_list[4])
        #     target.ext|end([class_id, center_x, center_y, width, height])
        info = object_infos[0].strip().split(" ")
        class_id = float(info[0])
        center_x = float(info[1])
        center_y = float(info[2])
        width = float(info[3])
        height = float(info[4])

        target = torch.tensor(target)
        target[0] = center_x
        target[1] = center_y
        target[2] = width
        target[3] = height
        # one-hot 编码：第 4+class_id 位置设为1
        target[4 + int(class_id)] = 1
        if self.transform is not None:
            image = self.transform(image)
        return image, target

def main():
    yolo_dataset = YOLODATASET(r"D:\code\yolo\ultralytics-main\datasets\bvn\images\train",
                               r"D:\code\yolo\ultralytics-main\datasets\bvn\labels\train",
                               transforms.Compose([transforms.ToTensor()]),
                               None)
    print(len(yolo_dataset))
    image, target = yolo_dataset[11]
    print(target)

if __name__ == '__main__':
    main()
