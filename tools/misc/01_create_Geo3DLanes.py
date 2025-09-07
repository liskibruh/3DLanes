import os
import sys
import glob
import shutil
import logging
import argparse
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile
from itertools import repeat
from multiprocessing import Pool # or ThreadPool?

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reformats ONCE and ONCE3DLanes Annotations to standard mmdet format")
    parser.add_argument(
        '--once-path',
        type=str,
        help='path to the ONCE annotation file',
        default='../../data/ONCE'
    )
    parser.add_argument(
        '--once3d-path',
        type=str,
        help='path to the ONCE3DLanes annotation file',
        default='../../data/ONCE3DLanes'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        help='where to save the reformatted annotation files',
        default='../../data/Geo3DLanes'
    )
    parser.add_argument(
        '--workers',
        type=int,
        help='num of workers for multiprocessing Pool',
        default=int(os.cpu_count())-2,
    )
    args = parser.parse_args()
    return args

def get_valid_scenes(once_path: str, once3d_path: str) -> tuple[list[str], list[str]]:
    def extract_data(once_path: str):
        assert isinstance(once_path, str), f"'once_path' should be of type str, but it is of type {type(once_path)}"
        assert isinstance(once3d_path, str), f"'once3d_path' should be a type str, but it is of type {type(once3d_path)}"

        if not os.path.exists(once_path):
            logger.critical(f'The dir {once_path} does not exist. Terminating!')
            sys.exit(1)
            
        if not os.path.exists(once3d_path):
            logger.critical(f'The dir {once3d_path} does not exist. Terminating!')
            sys.exit(1)

        for split in ['train', 'val']:
            if os.path.exists(os.path.join(once_path, split)):
                for obj in os.listdir(os.path.join(once_path, split)):
                    if obj.endswith(".tar"):
                        file = os.path.join(once_path, split, obj)
                        logger.info(f'Unzipping {file}')
                        TarFile(file).extractall(path=os.path.join(once_path, split))
                    elif obj.endswith(".zip"):
                        file = os.path.join(once_path, split, obj)
                        logger.info(f'Unzipping {file}')
                        ZipFile(file).extractall(path=os.path.join(once_path, split))
            else:
                logger.warning(f'{split} split dir not found. Processing remaining splits.')
                continue
    
    extract_data(once_path) # unzip ONCE data dirs [UNCOMMENT THIS IF FOUND COMMENTED]

    once_train_scenes, once3d_train_scenes = list(), list()
    once_val_scenes, once3d_val_scenes = list(), list()
    for split in ['train', 'val']:
        if split == 'train':
            once_train_scenes = [f for f in os.listdir(os.path.join(once_path, split, 'data')) 
                                 if os.path.isdir(os.path.join(once_path, split, 'data', f))]
            once3d_train_scenes = [f for f in os.listdir(os.path.join(once3d_path, split)) 
                                   if os.path.isdir(os.path.join(once3d_path, split, f))]
        elif split == 'val':
            if os.path.exists(os.path.join(once_path, split))\
                and os.path.exists(os.path.join(once3d_path, split)):
                once_val_scenes = [f for f in os.listdir(os.path.join(once_path, split, 'data')) 
                                if os.path.isdir(os.path.join(once_path, split, 'data', f))]
                once3d_val_scenes = [f for f in os.listdir(os.path.join(once3d_path, split)) 
                                    if os.path.isdir(os.path.join(once3d_path, split, f))]
            else:
                logger.warning(f"Can't find valid val scenes becasue {split} split does not exist either in ONCE or ONCE3D. Skipping split..")

    valid_train_scenes = list(set(once_train_scenes) & set(once3d_train_scenes))
    valid_val_scenes = list(set(once_val_scenes) & set(once3d_val_scenes))

    return valid_train_scenes, valid_val_scenes

def make_geo3dlanes(valid_scene: str, once_path: str, once3d_path: str, geo3d_path: str):
    for split in ['train', 'val']:
        os.makedirs(os.path.join(geo3d_path, split), exist_ok=True)

        once_split = os.path.join(once_path, split)
        once3d_split = os.path.join(once3d_path, split)
        geo3d_split = os.path.join(geo3d_path, split)

        if not (os.path.exists(once_split) and os.path.exists(once3d_split)):
            logger.warning(f"split {split} not found in either ONCE or ONCE3DLanes or both. Skipping split.")
            continue

        once_scenes = [str(f) for f in os.listdir(os.path.join(once_split, 'data')) 
                       if os.path.isdir(os.path.join(once_split, 'data', f))]
        once3d_scenes = [str(f) for f in os.listdir(once3d_split) 
                         if os.path.isdir(os.path.join(once3d_split, f))]

        if valid_scene in once_scenes and valid_scene in once3d_scenes:
            im_src = os.path.join(once_split, 'data', valid_scene, 'cam01')
            lidar_src = os.path.join(once_split, 'data', valid_scene, 'lidar_roof')
            bbox_annos_src = os.path.join(once_split, 'data', valid_scene, valid_scene+".json")
            lane_annos_src = os.path.join(once3d_split, valid_scene)

            scene_dst = os.path.join(geo3d_split, valid_scene)

            logger.info(f"Copying {valid_scene} -> {scene_dst}")
            shutil.copytree(im_src, os.path.join(scene_dst, 'cam01'), dirs_exist_ok=True)
            shutil.copytree(lidar_src, os.path.join(scene_dst, 'lidar_roof'), dirs_exist_ok=True)
            shutil.copy2(bbox_annos_src, scene_dst)
            shutil.copytree(lane_annos_src, os.path.join(scene_dst, 'lanes'), dirs_exist_ok=True)
        else:
            logger.warning(f"Scene {valid_scene} not found in either ONCE or ONCE3DLanes. Skipping scene {valid_scene}")

def load_timestamps():
    pass

def make_geo3dlanes_new(valid_scene: str, once_path: str, once3d_path: str, geo3d_path: str):
    for split in ['train', 'val']:
        os.makedirs(os.path.join(geo3d_path, split), exist_ok=True)
        once3d_scene = os.path.join(once3d_path, split, valid_scene)
        once_scene = os.path.join(once_path, split, 'data', valid_scene)

        if not (os.path.exists(os.path.join(once_path, split)) 
                and os.path.exists(os.path.join(once3d_path, split))):
            logger.warning(f"Split '{split}' not found in either ONCE or ONCE3DLanes or both. Skipping split.")
            continue

        # copy lane annotations and images
        fnames = os.listdir(os.path.join(once3d_scene, 'cam01'))

        logger.info(f"Copying Lane Annotations and Images to {geo3d_path}")
        for fname in fnames:
            ann_src = os.path.join(once3d_scene, 'cam01', fname)
            ann_dst = os.path.join(geo3d_path, split, valid_scene, 'lanes')
            os.makedirs(ann_dst, exist_ok=True)
            shutil.copy2(ann_src, ann_dst)

            file, _ = os.path.splitext(fname)
            im_src = os.path.join(once_scene, 'cam01', file) + '.jpg'
            im_dst = os.path.join(geo3d_path, split, valid_scene, 'cam01')
            os.makedirs(im_dst, exist_ok=True)
            shutil.copy2(im_src, im_dst)

        # copy lidar scans and object annotations
        logger.info(f"Copying LiDAR PointClouds to {geo3d_path}")
        lidar_src = os.path.join(once_scene, 'lidar_roof')
        lidar_dst = os.path.join(geo3d_path, split, valid_scene, 'lidar_roof')
        shutil.copytree(lidar_src, lidar_dst, dirs_exist_ok=True)
        
        logger.info(f"Copying Object Annotations")
        obj_anno_src = os.path.join(once_scene, f"{valid_scene}.json")
        obj_anno_dst = os.path.join(geo3d_path, split, valid_scene)
        shutil.copy2(obj_anno_src, obj_anno_dst)


def main():
    # logger configuration
    logging.basicConfig(filename='./make_Geo3DLanes.log', 
                        filemode='w',
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%a, %d %b %Y %H:%M:%S +0000",
                        level=logging.INFO,)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    
    # args and vars
    args = parse_args()
    once_path = str(args.once_path)
    once3d_path = str(args.once3d_path)
    save_dir = str(args.save_dir)
    workers = int(args.workers)

    # body
    valid_train_scenes, valid_val_scenes = get_valid_scenes(once_path, once3d_path)
    valid_scenes = valid_train_scenes + valid_val_scenes
    if len(valid_scenes)>0:
        if workers>1:
            with Pool(processes=workers) as pool:
                pool.starmap(make_geo3dlanes_new, 
                            zip(
                                valid_scenes,
                                repeat(once_path),
                                repeat(once3d_path),
                                repeat(save_dir))
                                )
        else:
            for scene in valid_scenes:
                make_geo3dlanes_new(scene, once_path, once3d_path, save_dir)

if __name__ == "__main__":
    main()