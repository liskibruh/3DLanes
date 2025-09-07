import os
import json
import pickle as pkl
from tqdm import tqdm
from itertools import repeat
from multiprocessing import Pool

def get_context_pcs(curr_pc: str, history=1):
    pass

def get_sensor_params(obj_ann_pth: str, frame_id=None):
    with open(obj_ann_pth, 'r') as infile:
        data = json.load(infile)    
    if frame_id is None:
        calib = data['calib']['cam01'] # only front cam calibs, todo: clibs should be in metainfo, not with every sample
        return calib
    else:
        for frame in data['frames']:
            if frame['frame_id'] == frame_id:
                pose = frame['pose']
        return pose

def get_lane_instances(lane_ann_pth: str, frame_id: str):
    if not os.path.exists(lane_ann_pth):
        raise FileNotFoundError(f"File {lane_ann_pth} could not be found!")
    
    with open(lane_ann_pth, 'r') as infile:
        data = json.load(infile)
    lanes = data['lanes']
    lane_num = data['lane_num']
    inst_dict = {'lane_num': data['lane_num'], 'lane_label': 0, 'lanes': data['lanes']}
    return inst_dict

def parse_sample(im_pth: str, lane_ann_pth: str, scene: str,  obj_ann_pth: str, pc_pth: str, split_pth: str, out_dict: dict):
    if any (not isinstance(p, str) 
            for p in [im_pth, lane_ann_pth, scene, obj_ann_pth, pc_pth, split_pth]):
        raise TypeError("All paths should be of type str")

    frame_id, _ = os.path.splitext(im_pth.split("/")[-1])

    sample_data = {'scene_id': scene,
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

    out_dict.setdefault('data_list', []).append(sample_data)

def create_pkl(scene: str, split_pth: str):
    print("processing scene")
# create train ann file in the following format
# {
#   "metainfo": {
#     "classes": ["lane"],
#     "lidar_sweeps": 5,          # number of sweeps per sample
#     "use_pose": true
#   },
#   "data_list": [
#     {
#       "sample_id": "000123",
#       "img_path": "train/images/000123.jpg",   # optional, if you use camera
#       "height": 1080,
#       "width": 1920,

#       "lidar_points": [
#         "train/lidar/000123.bin",
#         "train/lidar/000122.bin",
#         "train/lidar/000121.bin",
#         "train/lidar/000120.bin",
#         "train/lidar/000119.bin"
#       ],

#       "pose": [
#         [1, 0, 0, 1.2],
#         [0, 1, 0, -0.3],
#         [0, 0, 1, 0.8],
#         [0, 0, 0, 1]
#       ],

#       "cam_to_velo": {
#         "front": [[...], [...], [...], [...]],
#         "left": [[...], [...], [...], [...]],
#         "right": [[...], [...], [...], [...]]
#       },

#       "instances": [
#         {
#           "lane_3d": [
#             [x1, y1, z1],
#             [x2, y2, z2],
#             [x3, y3, z3],
#             ...
#           ],
#           "lane_label": 0,
#           "ignore_flag": 0
#         }
#       ]
#     }
#   ]
# }

    # sample names are taken from lanes because we only want ONCE samples that has lane annotations
    samples = []
    for s in os.listdir(os.path.join(split_pth, scene, 'lanes')):
        s.strip()
        sname, _ = os.path.splitext(s)
        samples.append(sname)
    
    out_dict = dict()
    out_dict['data_list'] = []
    for s in samples:
        im_pth = os.path.join(split_pth, scene, 'cam01', s + '.jpg')
        lane_ann_pth = os.path.join(split_pth, scene, 'lanes', s + '.json')
        obj_ann_pth = os.path.join(split_pth, scene, scene+'.json')
        pc_pth = os.path.join(split_pth, scene, 'lidar_roof', s + '.bin')


        # pseudo-code after this point (todo: write the finalized code)
        parse_sample(im_pth, lane_ann_pth, scene, obj_ann_pth, pc_pth, split_pth, out_dict)

    return out_dict


if __name__ == "__main__":
    data_root = "/data24t_1/owais.tahir/3DLanes/mmdetection/data/Geo3DLanes"
    out_ann_pth = os.path.join(data_root, "geo3dlanes.pkl")

    workers = max(os.cpu_count()//2, 1)

    for split in ['train', 'val']:
        split_pth = os.path.join(data_root, split)
        scenes = os.listdir(split_pth)
        
        if workers > 1:
            with Pool(processes=workers) as pool:
                results = pool.starmap(create_pkl, list(zip(scenes, repeat(split_pth))))
        else:
            results = []
            for scene in tqdm(scenes, total=len(scenes), desc=("Processing scenes")):
                results.append(create_pkl(scene, split_pth))

    merged = {'metainfo': {'classes': ['lane'], 'lidar_sweeps': 2}, 'data_list': []}
    for r in results:
        merged['data_list'].extend(r['data_list'])

    with open(out_ann_pth, 'wb') as ofile:
        pkl.dump(merged, ofile)