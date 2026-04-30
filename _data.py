import configparser
import glob
import os
import os.path as osp
import pickle
import platform
from pathlib import Path
from random import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler, Sampler
from torchvision import transforms as T



def get_class_num(name):
    r = {"cifar": 10, "flickr": 38, "nuswide": 21, "coco": 80,"imagenet":100,"things": 200}[name]
    return r


def get_topk(name):
    r = {"cifar": None, "flickr": None, "nuswide": 5000, "coco": None,"imagenet":None,"things": None,"things":None}[name]
    return r


def get_concepts(name, root):
    with open(osp.join(root, name, "concepts.txt"), "r") as f:
        lines = f.read().splitlines()
    return np.array(lines)


def build_trans(usage, resize_size=256, crop_size=224):
    if usage == "train":
        steps = [T.RandomCrop(crop_size), T.RandomHorizontalFlip()]
    else:
        steps = [T.CenterCrop(crop_size)]
    return T.Compose(
        [T.Resize(resize_size)]
        + steps
        + [
            T.ToTensor(),
            # T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

def build_loaders(name, root, **kwargs):

    train_trans = build_trans("train")
    other_trans = build_trans("other")

    data = init_dataset(name, root)
    # 统一标签维度
    max_classes = data.train[0][1].shape[0]
    for subset in [data.query, data.dbase]:
        for i in range(len(subset)):
            img, lab = subset[i]
            if lab.shape[0] < max_classes:
                lab = np.pad(lab, (0, max_classes - lab.shape[0]), 'constant')
                subset[i] = (img, lab)

    train_loader = DataLoader(ImageDataset(data.train, train_trans), shuffle=True, drop_last=True, **kwargs)
    query_loader = DataLoader(ImageDataset(data.query, other_trans), **kwargs)
    dbase_loader = DataLoader(ImageDataset(data.dbase, other_trans), **kwargs)

    return train_loader, query_loader, dbase_loader



class BaseDataset(object):
    """
    Base class of dataset
    """

    def __init__(self, name, txt_root, img_root, verbose=True):
        self.name = name  # 添加这行代码保存数据集名称
        # self.num_classes = get_class_num(name)  # 在基类中初始化类别数
        self.img_root = img_root
        self.verbose=verbose
        self.train_txt = osp.join(txt_root, "train.txt")
        self.query_txt = osp.join(txt_root, "query.txt")
        self.dbase_txt = osp.join(txt_root, "dbase.txt")

        self.check_before_run()

        self.load_data()

        self.train = self.process(self.train_txt)
        self.query = self.process(self.query_txt)
        self.dbase = self.process(self.dbase_txt)

        self.unload_data()

        if verbose:
            print(f"=> {name.upper()} loaded")
            self.print_dataset_statistics(self.train, self.query, self.dbase)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_txt):
            raise RuntimeError("'{}' is not available".format(self.train_txt))
        if not osp.exists(self.query_txt):
            raise RuntimeError("'{}' is not available".format(self.query_txt))
        if not osp.exists(self.dbase_txt):
            raise RuntimeError("'{}' is not available".format(self.dbase_txt))

    def get_imagedata_info(self, data):
        if not data:
            return 0, 0

        # 获取第一个样本的标签，判断是否为多标签
        first_lab = data[0][1]
        if isinstance(first_lab, np.ndarray) and first_lab.ndim == 1 and len(first_lab) > 1:
            # 多标签逻辑：统计有标注的类别数
            labs = np.array([lab for _, lab in data])
            n_cids = (labs.sum(axis=0) > 0).sum()
        else:
            # 单标签逻辑：提取标量标签统计唯一值
            labs = np.array([
                lab[0] if isinstance(lab, np.ndarray) else lab
                for _, lab in data
            ])
            n_cids = len(np.unique(labs))

        n_imgs = len(data)
        return n_cids, n_imgs

    def print_dataset_statistics(self, train, query, dbase):
        n_train_cids, n_train_imgs = self.get_imagedata_info(train)
        n_query_cids, n_query_imgs = self.get_imagedata_info(query)
        n_dbase_cids, n_dbase_imgs = self.get_imagedata_info(dbase)
        # 打印标签形状
        train_labels = np.array([lab for _, lab in train])
        query_labels = np.array([lab for _, lab in query])
        dbase_labels = np.array([lab for _, lab in dbase])
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Query labels shape: {query_labels.shape}")
        print(f"Dbase labels shape: {dbase_labels.shape}")

        # # 统一使用训练集的类别数
        # train_labels = np.array([lab for _, lab in train])
        # train_classes = (train_labels.sum(axis=0) > 0)  # bool数组
        # n_train_cids = train_classes.sum()  # 训练集的类别数
        #
        # n_train_imgs = len(train)
        # n_query_imgs = len(query)
        # n_dbase_imgs = len(dbase)

        # print(f"Train labels shape: {train_labels.shape}")
        # print(f"Query labels shape: {np.array([lab for _, lab in query]).shape}")
        # print(f"Dbase labels shape: {np.array([lab for _, lab in dbase]).shape}")

        print("Image Dataset statistics:")
        print("  -----------------------------")
        print("  subset | # images | # classes")
        print("  -----------------------------")
        print("  train  | {:8d} | {:9d}".format(n_train_imgs, n_train_cids))
        print("  query  | {:8d} | {:9d}".format(n_query_imgs, n_query_cids))
        print("  dbase  | {:8d} | {:9d}".format(n_dbase_imgs, n_dbase_cids))
        print("  -----------------------------")

    def load_data(self):
        pass

    def unload_data(self):
        pass

    def process(self, txt_path):
        dataset = [
            (
                osp.join(self.img_root, x.split()[0]),
                np.array(x.split()[1:], dtype=np.float32),
            )
            for x in open(txt_path, "r").readlines()
        ]
        return dataset


class CIFAR(BaseDataset):

    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)

    @staticmethod
    def unpickle(file):
        with open(file, "rb") as fo:
            dic = pickle.load(fo, encoding="latin1")
        return dic

    def load_data(self):
        data_list = [f"data_batch_{x}" for x in range(1, 5 + 1)]
        data_list.append("test_batch")
        imgs = []
        for x in data_list:
            data = self.unpickle(osp.join(self.img_root, x))
            imgs.append(data["data"])
            # labs.extend(data["labels"])
        imgs = np.vstack(imgs).reshape(-1, 3, 32, 32)
        self.imgs = imgs.transpose((0, 2, 3, 1))

    def process(self, txt_path):
        dataset = []
        for x in open(txt_path, "r").readlines():
            idx = int(x.split()[0].replace(".png", ""))
            # lab1 = np.squeeze(np.eye(10, dtype=np.float32)[self.labs[idx]])
            lab = np.array(x.split()[1:], dtype=np.float32)
            dataset.append((self.imgs[idx], lab))
        return dataset

    def unload_data(self):
        self.imgs = None


class NUSWIDE(BaseDataset):

    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)

    def load_data(self):
        self.imgs = {p.stem: str(p) for p in Path(self.img_root).rglob("*.jpg")}

    def process(self, txt_path):
        dataset = []
        for x in open(txt_path, "r").readlines():
            key = x.split()[0].replace(".jpg", "")
            lab = np.array(x.split()[1:], dtype=np.float32)
            dataset.append((self.imgs[key], lab))
        return dataset

    def unload_data(self):
        self.imgs.clear()


class COCO(NUSWIDE):

    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)

    def load_data(self):
        self.imgs = {p.stem.split("_")[-1]: str(p) for p in Path(self.img_root).rglob("*.jpg")}


class ImageNet(BaseDataset):
    def __init__(self, name, txt_root, img_root, verbose=True):
        # 先初始化子类属性，再调用父类构造函数
        self.imgs = {}  # 存储 {文件名: 完整路径} 的映射
        self.dataset = []
        super().__init__(name, txt_root, img_root, verbose)

    def load_data(self):
        """重新构建图像路径映射，确保包含train/类别文件夹层级"""
        # 1. 处理训练集和数据库集图像（train/类别文件夹/文件名）
        train_pattern = os.path.join(self.img_root, "train", "*", "*.JPEG")  # *匹配类别文件夹
        train_img_paths = glob.glob(train_pattern)
        for path in train_img_paths:
            filename = os.path.basename(path)  # 提取文件名（如n03838899_30611.JPEG）
            self.imgs[filename] = path  # 存储完整路径

        # 2. 处理验证集图像（val/文件名）
        val_pattern = os.path.join(self.img_root, "val", "*.JPEG")
        val_img_paths = glob.glob(val_pattern)
        for path in val_img_paths:
            filename = os.path.basename(path)
            self.imgs[filename] = path

        if self.verbose:
            print(f"Loaded {len(self.imgs)} images (train: {len(train_img_paths)}, val: {len(val_img_paths)})")

    def process(self, txt_path):
        """严格根据文件名匹配路径，不依赖推断"""
        dataset = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                print(f"跳过无效行：{line}")
                continue

            filename = parts[0]  # 如n03838899_30611.JPEG
            label = np.array([int(p) for p in parts[1:]], dtype=np.float32)

            # 直接从预加载的路径映射中查找
            if filename in self.imgs:
                img_path = self.imgs[filename]
                # 双重验证路径是否存在（避免缓存错误）
                if os.path.exists(img_path):
                    dataset.append((img_path, label))
                else:
                    print(f"路径存在但文件无效：{img_path}")
            else:
                # 明确报错，不猜测路径
                print(f"严重错误：{filename} 不在图像列表中（请检查图像路径是否正确）")

        print(f"从{txt_path}处理了{len(dataset)}个样本")
        return dataset

    # def __getitem__(self, idx):
    #     """加载图像时增加路径检查"""
    #     img_path, label = self.dataset[idx]
    #     try:
    #         if not os.path.exists(img_path):
    #             raise FileNotFoundError(f"图像文件不存在：{img_path}")
    #         img = Image.open(img_path).convert("RGB")
    #         if self.transform:
    #             img = self.transform(img)
    #         return img, label
    #     except Exception as e:
    #         print(f"加载图像失败：{e}")
    #         # 返回占位符避免程序崩溃
    #         return Image.new('RGB', (224, 224)), np.zeros_like(label)


class THINGS(BaseDataset):
    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)
        # self.filter_labels_by_train_classes()
    def load_data(self):
        self.imgs = {}
        exts = ('*.jpg', '*.jpeg', '*.png')
        for ext in exts:
            for p in Path(self.img_root).rglob(ext):
                # 用相对路径，不包含最外层的images文件夹
                rel_path = os.path.relpath(str(p), self.img_root).replace('\\', '/')
                # 如果开头是 images/，去掉它，方便匹配
                if rel_path.startswith("images/"):
                    rel_path = rel_path[len("images/"):]
                self.imgs[rel_path] = str(p)
        if self.verbose:
            print(f"[THINGS] Loaded {len(self.imgs)} images from {self.img_root}")

    def process(self, txt_path):
        dataset = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # 这里也去除 images/ 前缀，和load_data一致
            img_path = parts[0].replace('\\', '/')
            if img_path.startswith("images/"):
                img_path = img_path[len("images/"):]
            label = np.array([float(x) for x in parts[1:]], dtype=np.float32)

            # if img_path in self.imgs:
            #     abs_path = self.imgs[img_path]
            #     dataset.append((abs_path, label))
            # else:
            #     # print(f"[THINGS Warning] Image path not found: {img_path}")
            #     print(f"")
            if img_path in self.imgs:
                abs_path = self.imgs[img_path]
                try:
                    # 这里尝试打开图片，确保有效
                    with Image.open(abs_path) as img:
                        img.verify()  # PIL 专门检测图片完整性的函数
                    dataset.append((abs_path, label))
                except Exception as e:
                    print(f"[THINGS Warning] Bad image skipped: {abs_path}, error: {e}")
            else:
                print(f"[THINGS Warning] Image path not found: {img_path}")

        if self.verbose:
            print(f"[THINGS] Processed {len(dataset)} samples from {txt_path}")
        return dataset

    def __getitem__(self, idx):
        img, lab = self.data[idx]
        if isinstance(img, str):
            try:
                img = Image.open(img).convert("RGB")
            except Exception as e:
                print(f"[Warning] Bad image skipped: {img}, error: {e}")
                # 返回一张纯白占位图
                img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, lab, idx

    def filter_labels_by_train_classes(self):
        # 训练集出现的类别
        train_labels = np.array([lab for _, lab in self.train])
        train_classes = (train_labels.sum(axis=0) > 0)  # bool数组

        def filter_sample(sample):
            img, lab = sample
            lab_filtered = lab * train_classes
            return img, lab_filtered

        self.query = [filter_sample(s) for s in self.query]
        self.dbase = [filter_sample(s) for s in self.dbase]

        def not_all_zero(sample):
            return np.any(sample[1])

        self.query = [s for s in self.query if not_all_zero(s)]
        self.dbase = [s for s in self.dbase if not_all_zero(s)]

    def unload_data(self):
        self.imgs.clear()


_ds_factory = {"cifar": CIFAR, "nuswide": NUSWIDE, "flickr": BaseDataset, "coco": COCO, "imagenet": ImageNet,
               "things": THINGS,"things_top100":THINGS,"things_top1000":THINGS,"things_top500":THINGS,"things_top200":THINGS,}

def init_dataset(name, root, **kwargs):
    if name not in list(_ds_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(_ds_factory.keys())))

    txt_root = osp.join(root, name)

    ini_loc = osp.join(root, name, "images", "location.ini")
    if osp.exists(ini_loc):
        config = configparser.ConfigParser()
        config.read(ini_loc)
        img_root = config["DEFAULT"][platform.system()]
    else:
        img_root = osp.join(root, name)
    print(f"txt_root:{txt_root}")
    print(f"img_root:{img_root}")

    dataset = _ds_factory[name](name, txt_root, img_root, **kwargs)


    return dataset


class ImageDataset(Dataset):
    """Image Dataset"""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     img, lab = self.data[idx]
    #     if isinstance(img, str):
    #         img = Image.open(img).convert("RGB")
    #     else:
    #         img = Image.fromarray(img)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     return img, lab, idx


    # def __getitem__(self, idx):
    #     img_path, lab = self.data[idx]
    #     try:
    #         if isinstance(img_path, str):
    #             img = Image.open(img_path).convert("RGB")
    #         else:
    #             img = Image.fromarray(img_path)
    #     except Exception as e:
    #         print(f"[Warning] Failed to load image: {img_path}, error: {e}")
    #         # 返回占位符：黑色图片 + 全 0 标签
    #         img = Image.new('RGB', (224, 224))
    #         lab = np.zeros_like(lab)
    #
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     return img, lab, idx

    def __getitem__(self, idx):
        img_path, lab = self.data[idx]
        try:
            if isinstance(img_path, str):
                img = Image.open(img_path).convert("RGB")
            else:
                img = Image.fromarray(img_path)
        except Exception as e:
            print(f"[Warning] Bad image skipped: {img_path}, error: {e}")
            print(f"")
            img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        if self.transform is not None:
            img = self.transform(img)
        return img, lab, idx

    def get_all_labels(self):
        return torch.from_numpy(np.vstack([x[1] for x in self.data]))


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # dataset = init_dataset("things_top200", "./_datasets")
    # dataset = init_dataset("things_top1000", "./_datasets")
    # dataset = init_dataset("things_top100", "./_datasets")
    dataset = init_dataset("imagenet", "./_datasets")
    trans = T.Compose(
        [
            # T.ToPILImage(),
            T.Resize([224, 224]),
            T.ToTensor(),
        ]
    )

    train_set = ImageDataset(dataset.train, trans)

    dataloader = DataLoader(train_set, batch_size=1, shuffle=True)

    for images, labels, _ in dataloader:
        print(images.shape, labels)
        plt.imshow(images[0].numpy().transpose(1, 2, 0))
        title = labels[0].argmax().item()
        print(title)
        plt.title(title)
        plt.show()
        break

