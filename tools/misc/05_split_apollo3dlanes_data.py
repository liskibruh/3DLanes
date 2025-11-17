import os
import json
import random

def get_lanes_data(label_fpath: str) -> dict:
    out_dict = {}
    with open(label_fpath, 'r') as infile:
        data = json.load(infile)

    lanes_data = data['laneBoundaryList'] # list    
    ids, labels, colors, widths = list(), list(), list(), list()
    lanes3d = []
    for s in lanes_data:
        lane = [[point['x'], point['y'], point['z']] for point in s['pos3DInCameraList']]
        lanes3d.append(lane)
        ids.append(s['id'])
        labels.append(s['type'])
        colors.append(s['color'])
        widths.append(s['width'])
    cam_pitch = data['cameraPitch']
    cam_height = data['cameraHeight']
        
    return {"ids": ids,
            "num_lanes": len(lanes3d),
            "lanes": lanes3d, 
            "labels": labels, 
            "colors": colors, 
            "widths": widths,
            "cam_pitch": cam_pitch,
            "cam_height": cam_height}

if __name__ == "__main__":
    data_root = "../../data/Apollo_Sim_3D_Lane_Release"
    labels_dir = os.path.join(data_root, 'labels')
    out_dir = os.path.join(data_root, "data_splits", "lanes_in_cam")
    os.makedirs(out_dir, exist_ok=True)
    scenes = os.listdir(labels_dir)
    all_annos = []

    print("Extracting data..")
    for scene in scenes:
        scene_dir = os.path.join(labels_dir, scene)
        label_files = os.listdir(scene_dir)
        
        for label in label_files:
            label_fpath = os.path.join(os.path.join(scene_dir, label))
            lanes_data = get_lanes_data(label_fpath)
            lanes_data["img_prefix"] = os.path.join(data_root, "images", scene) + "/"
            lanes_data["img_path"] = "/".join(label_fpath.strip().split('/')[-2: ]).split(".")[0] + '.jpg' # sorry for this mess
            all_annos.append(lanes_data)

    random.shuffle(all_annos)
    split_ratios = [0.85, 0.15] # 85% train, 15% test/val
    N = len(all_annos)
    N1 = int(N * split_ratios[0])
    N2 = int(N * split_ratios[1])
    print(f"No. of train samples: {N1}")
    print(f"No. of val samples: {N2}")

    train_split = all_annos[0: N1]
    test_split = all_annos[N1: N1+N2]

    print("Writing data..")
    for anno in train_split:
        out_file = out_dir + "/train.json"
        with open(out_file, 'a') as f:
            f.write(json.dumps(anno))
            f.write("\n")
            
    for anno in test_split:
        out_file = out_dir + "/val.json"
        with open(out_file, 'a') as f:
            f.write(json.dumps(anno))
            f.write("\n")

    print("DONE!")