from utils import read_labels_to_file, extract_frames


local_path = '/Users/seanxiang/data/cholec80/'
remote_path = '/home/cxia8134/data/'

fps = 25

# read_labels_to_file(remote_path, fps, 1, 41, 'train')
# read_labels_to_file(remote_path, fps, 41, 51, 'vali')

for i in range(51, 61):
    read_labels_to_file(local_path, fps, i, i+1, 'test')
