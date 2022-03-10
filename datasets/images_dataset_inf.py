from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import pickle


class ImagesDataset(Dataset):

	def __init__(self, x, no_gd, with_gd, opts, target_transform=None, source_transform=None):
		self.x_path = sorted(data_utils.make_dataset(x))
		self.no_gd_path = sorted(data_utils.make_dataset(no_gd))
		self.with_gd_path = sorted(data_utils.make_dataset(with_gd))
    
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
  
		with open("./eth_label_dict.pickle", "rb") as f:
			label_dict = pickle.load(f)
		self.label = label_dict


	def __len__(self):
		return len(self.x_path)

	# FROM IMAGE, TO IMAGE, GAZE LABEL
	def __getitem__(self, index):
		x_path = self.x_path[index]
		x_img = Image.open(x_path)
		x_img = x_img.convert('RGB')
  
		no_gd_path = self.no_gd_path[index]
		no_gd_img = Image.open(no_gd_path)
		no_gd_img = no_gd_img.convert('RGB')

		with_gd_path = self.with_gd_path[index]
		with_gd_img = Image.open(with_gd_path)
		with_gd_img = with_gd_img.convert('RGB')
  
		# print("from path: ", from_path)
		
		img_name = x_path.split("/")[-1]
		img_name = img_name.split(".")[0]

		# print(img_name)
		labels = self.label[img_name]
		
  # if self.target_transform:
		# 	to_im = self.target_transform(to_im)

		# if self.source_transform:
		# 	from_im = self.source_transform(from_im)
		# else:
		# 	from_im = to_im

		return x_img, no_gd_img, with_gd_img, labels
