from modules.utils import read_labels_to_file, extract_frames


local_path = '/Users/seanxiang/data/cholec80/'
remote_path = '/home/cxia8134/data/'

fps = 25

# extract_frames(remote_path, fps)
# read_labels_to_file(remote_path, fps, 1, 41, 'train')
# read_labels_to_file(remote_path, fps, 41, 51, 'vali')

# generate train (image path, label) pair
for i in range(1, 51):
    read_labels_to_file(remote_path, fps, i, i+1, 'train')

# generate vali pair
for i in range(51, 61):
    read_labels_to_file(remote_path, fps, i, i+1, 'vali')

# generate test pair
for i in range(61, 81):
    read_labels_to_file(remote_path, fps, i, i+1, 'test')
