import os
import json
import pickle as pkl
from tqdm import tqdm
from itertools import repeat
from multiprocessing import Pool

# -------------------- Helper Functions -------------------- #

def get_sensor_params(obj_ann_pth: str, frame_id=None):
    """Read sensor calibration and pose."""
    with open(obj_ann_pth, 'r') as infile:
        data = json.load(infile)
    
    if frame_id is None:
        return data['calib']['cam01']
    else:
        for frame in data['frames']:
            if frame['frame_id'] == frame_id:
                return frame['pose']
        return None

def get_lane_instances(lane_ann_pth: str, frame_id: str):
    """Read lane annotations for a frame."""
    if not os.path.exists(lane_ann_pth):
        raise FileNotFoundError(f"File {lane_ann_pth} could not be found!")
    
    with open(lane_ann_pth, 'r') as infile:
        data = json.load(infile)
    
    inst_dict = {
        'lane_num': data['lane_num'],
        'lane_label': 0,  # default label
        'lanes': data['lanes']
    }
    return inst_dict

def parse_sample(im_pth: str, lane_ann_pth: str, scene: str, obj_ann_pth: str, pc_pth: str, split_pth: str):
    """Construct a single sample dictionary."""
    frame_id, _ = os.path.splitext(os.path.basename(im_pth))
    
    sample_data = {
        'scene_id': scene,
        'frame_id': frame_id,
        'img_path': im_pth,
        'height': 1020,
        'width': 1920,
        'pose': get_sensor_params(obj_ann_pth, frame_id),
        'calib': get_sensor_params(obj_ann_pth),
        'lane_ann_pth': lane_ann_pth,
        'obj_ann_pth': obj_ann_pth,
        'pc_pth': pc_pth,
        'instances': get_lane_instances(lane_ann_pth, frame_id)
    }
    return sample_data

def create_pkl(scene: str, split_pth: str):
    """Create data_list for one scene."""
    lane_folder = os.path.join(split_pth, scene, 'lanes')
    if not os.path.exists(lane_folder):
        return {'data_list': []}

    samples = [os.path.splitext(f)[0] for f in os.listdir(lane_folder)]

    out_list = []
    for s in samples:
        im_pth = os.path.join(split_pth, scene, 'cam01', s + '.jpg')
        lane_ann_pth = os.path.join(split_pth, scene, 'lanes', s + '.json')
        obj_ann_pth = os.path.join(split_pth, scene, scene + '.json')
        pc_pth = os.path.join(split_pth, scene, 'lidar_roof', s + '.bin')

        # parse sample
        sample_data = parse_sample(im_pth, lane_ann_pth, scene, obj_ann_pth, pc_pth, split_pth)
        out_list.append(sample_data)

    return {'data_list': out_list}

# wrapper for multiprocessing
def create_pkl_wrapper(args):
    return create_pkl(*args)

# -------------------- Main -------------------- #

if __name__ == "__main__":
    data_root = "/data24t_1/owais.tahir/3DLanes/mmdetection/data/Geo3DLanes"
    out_ann_pth = os.path.join(data_root, "geo3dlanes_train.pkl")

    workers = max(os.cpu_count() // 2, 1)  # safe default

    merged = {'metainfo': {'classes': ['lane'], 'lidar_sweeps': 2}, 'data_list': []}

    for split in ['train', 'val']:
        split_pth = os.path.join(data_root, split)
        scenes = os.listdir(split_pth)

        if workers > 1:
            tasks = list(zip(scenes, repeat(split_pth)))
            with Pool(processes=workers) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(create_pkl_wrapper, tasks),
                        total=len(tasks),
                        desc=f"Processing {split} scenes"
                    )
                )
        else:
            results = []
            for scene in tqdm(scenes, total=len(scenes), desc=f"Processing {split} scenes"):
                results.append(create_pkl(scene, split_pth))

        # merge data_list from all scenes
        for r in results:
            merged['data_list'].extend(r['data_list'])

    # save final pickle
    with open(out_ann_pth, 'wb') as ofile:
        pkl.dump(merged, ofile)

    print(f"Saved {len(merged['data_list'])} samples to {out_ann_pth}")
