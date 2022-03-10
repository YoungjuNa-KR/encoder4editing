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
 
	'mpii_train': '/home/vimlab/Downloads/eth_256_edit/train',
	'mpii_test': '/home/vimlab/Downloads/eth_256_edit/val',

	'eth_256_train' : '/home/vimlab/Downloads/face_resize_eth_train',
	'eth_256_val' : '/home/vimlab/Downloads/face_resize_eth_validation',
 
	'gaze_x' : '/home/vimlab/Downloads/eth_256_edit/val',
 	'gaze_no_gd' : '/home/vimlab/encoder4editing/new/experiment/eth_256/no_gd_test_edited_eth/logs/images',
 	'gaze_with_gd' : '/home/vimlab/encoder4editing/new/experiment/eth_256/gd_test_edited_eth/logs/images',
}

model_paths = {
	'stylegan_ffhq': './pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': './pretrained_models/model_ir_se50.pth',
	'shape_predictor': './pretrained_models/shape_predictor_68_face_landmarks.dat',
	'moco': './pretrained_models/moco_v2_800ep_pretrain.pt',
	'gaze': '/home/vimlab/encoder4editing/pretrained_models/eth_gaze_256_gd_loss.pt',

	# custom model
	'stylegan_mpii': './pretrained_models/eth_256_fid10.pt',
	'stylegan_eth' : './pretrained_models/eth_256_fid10.pt'
}
