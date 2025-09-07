import pickle as pkl

pkl_pth = "../../data/Geo3DLanes/geo3dlanes.pkl"

with open(pkl_pth, 'rb') as ifile:
    data = pkl.load(ifile)

print(f"data[metainfo]: \n{data['metainfo']}")
print(f"data[data_list][0]: \n{data['data_list'][0]}")