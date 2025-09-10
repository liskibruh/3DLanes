import os
import json
import pickle as pkl
from tqdm import tqdm
from itertools import repeat
from multiprocessing import Pool

def parse_sample(frame_id, im_pth, lane_ann_data, scene, obj_ann_data, pc_pth):
    """Construct a single sample dictionary using preloaded JSONs."""
    pose = None
    for frame in obj_ann_data['frames']:
        if frame['frame_id'] == frame_id:
            pose = frame['pose']
            break

    instances = list()
    for lane in lane_ann_data['lanes']:
        label = 0 # dummy label [0]. ONCE3DLanes provide no information about lane classes
        ignore_flag = False if len(lane) > 2 else True 
        instances.append({'lane': lane, 'label': label, 'ignore_flag': ignore_flag})

    sample_data = {
        'scene_id': scene,
        'frame_id': frame_id,
        'img_path': im_pth,
        'height': 1020,
        'width': 1920,
        'pose': pose,
        'calib': obj_ann_data['calib']['cam01'], 
        'lane_ann_pth': lane_ann_data['__path__'], 
        'obj_ann_pth': obj_ann_data['__path__'],
        'pc_pth': pc_pth,
        'instances': instances,
    }
    return sample_data


def create_pkl(scene: str, split_pth: str):
    """Create data_list for one scene with preloaded JSONs."""
    lane_folder = os.path.join(split_pth, scene, 'lanes')
    if not os.path.exists(lane_folder):
        return {'data_list': []}

    # preload object annotations (once per scene)
    obj_ann_pth = os.path.join(split_pth, scene, scene + '.json')
    with open(obj_ann_pth, 'r') as f:
        obj_ann_data = json.load(f)
    obj_ann_data['__path__'] = obj_ann_pth

    samples = [os.path.splitext(f)[0] for f in os.listdir(lane_folder)]

    out_list = []
    for s in samples:
        im_pth = os.path.join(split_pth, scene, 'cam01', s + '.jpg')
        lane_ann_pth = os.path.join(split_pth, scene, 'lanes', s + '.json')
        pc_pth = os.path.join(split_pth, scene, 'lidar_roof', s + '.bin')

        # preload lane annotation (once per frame)
        with open(lane_ann_pth, 'r') as f:
            lane_ann_data = json.load(f)
        lane_ann_data['__path__'] = lane_ann_pth

        # parse sample with preloaded JSONs
        sample_data = parse_sample(s, im_pth, lane_ann_data, scene, obj_ann_data, pc_pth)
        out_list.append(sample_data)

    return {'data_list': out_list}


# wrapper for multiprocessing
def create_pkl_wrapper(args):
    return create_pkl(*args)


if __name__ == "__main__":
    data_root = "/data24t_1/owais.tahir/3DLanes/mmdetection/data/Geo3DLanes"
    out_ann_pth = os.path.join(data_root, "geo3dlanes_train.pkl")

    workers = max(os.cpu_count() // 2, 1)

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

        for r in results:
            merged['data_list'].extend(r['data_list'])

    with open(out_ann_pth, 'wb') as ofile:
        pkl.dump(merged, ofile)

    print(f"Saved {len(merged['data_list'])} samples to {out_ann_pth}")