dataset_paths = {
	#  Face Datasets (In the paper: FFHQ - train, CelebAHQ - test)
	'ffhq': '',
	'celeba_test': '',

	#  Cars Dataset (In the paper: Stanford cars)
	'cars_train': '',
	'cars_test': '',

	#  Horse Dataset (In the paper: LSUN Horse)
	'horse_train': '',
	'horse_test': '',

	#  Church Dataset (In the paper: LSUN Church)
	'church_train': '',
	'church_test': '',

	#  Cats Dataset (In the paper: LSUN Cat)
	'cats_train': '',
	'cats_test': '',
 
	'mpii_train': '/home/vimlab/youngju/encoder4editing/data/eth_256/train',
	'mpii_test': '/home/vimlab/youngju/encoder4editing/data/eth_256/val',

	'eth_256_train' : '/home/vimlab/youngju/encoder4editing/data/eth_256/train',
	'eth_256_val' : '/home/vimlab/youngju/encoder4editing/data/eth_256/validation'
}

model_paths = {
	'stylegan_ffhq': './pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': './pretrained_models/model_ir_se50.pth',
	'shape_predictor': './pretrained_models/shape_predictor_68_face_landmarks.dat',
	'moco': './pretrained_models/moco_v2_800ep_pretrain.pt',
	'gaze': './pretrained_models/gaze_model.pt',
 
 
	# custom model
	'stylegan_mpii': './pretrained_models/network-stapshot-018200.pt',
	'stylegan_eth' : './pretrained_models/network-stapstho-018200.pt'
}
