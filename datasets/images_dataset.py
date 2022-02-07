from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import pickle


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
  
		with open("./eth_label_dict.pickle", "rb") as f:
			label_dict = pickle.load(f)
		self.label = label_dict


	def __len__(self):
		return len(self.source_paths)

	# FROM IMAGE, TO IMAGE, GAZE LABEL
	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')
		# print("from path: ", from_path)
		
		img_name = from_path.split("/")[-1]
		img_name = img_name.split(".")[0]
		# print(img_name)
		labels = self.label[img_name]
 
		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im, labels
