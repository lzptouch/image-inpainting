import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    # any 判断可迭代参数是否全部为False
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path_files):
    # 读取文件 读取文件夹 获取文件列表
    if path_files.find('.flist') != -1:
        paths, size = make_dataset_txt(path_files)
    else:
        paths, size = make_dataset_dir(path_files)
    return paths, size


def make_dataset_txt(files):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []
    with open(files) as f:
        paths = f.readlines()
    for path in paths:
        path = path.strip()
        img_paths.append(path)
        # img_paths = sorted(img_paths)
    return img_paths, len(img_paths)


def make_dataset_dir(dir):
    """
    :param dir: directory paths that store the image
    :return: image paths and sizes
    """
    img_paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img_paths.append(path)
                # img_paths = sorted(img_paths)
    return img_paths, len(img_paths)
