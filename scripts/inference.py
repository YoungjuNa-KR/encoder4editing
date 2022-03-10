import argparse
from tqdm import tqdm
import torch
import numpy as np
import sys
import os
import dlib

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from utils.alignment import align_face
from PIL import Image


def main(args):
    net, opts = setup_model(args.ckpt, device)
    # net, opts = setup_model('./pretrained_models/eth_256_fid10.pt', device)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)

    # Check if latents exist
    latents_file_path = os.path.join(args.save_dir, 'latents.pt')
    if os.path.exists(latents_file_path):
        latent_codes = torch.load(latents_file_path).to(device)
    else:
        latent_codes, img_names = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)
        # torch.save(latent_codes, latents_file_path)

    if not args.latents_only:
        generate_inversions(args, generator, latent_codes, img_names, is_cars=is_cars)


def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    align_function = None
    if args.align:
        align_function = run_alignment
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, n_images=None, is_cars=False):
    all_latents = []
    i = 0
    all_names = []
    tqdm_test = tqdm(data_loader, ncols=80)
    with torch.no_grad():
        for batch in tqdm_test:
            x, name = batch
            if n_images is not None and i > n_images:
                break
            
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
        
            name = list(name)
            for i in range(len(name)):
                name[i] = name[i].replace("'", '')
                name[i] = name[i].replace("(", '')
                name[i] = name[i].replace(")", '')
                name[i] = name[i].replace(",", '')
                all_names.append(name[i])
            i += len(latents)
            # print(name)
            # print('----')
            torch.save(latents, './eyediap_latent_pt/'+str(name)[2:-2]+'.pt')
    return torch.cat(all_latents), all_names



def save_image(img, save_dir, filename):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{filename}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)


@torch.no_grad()
def generate_inversions(args, g, latent_codes, img_names, is_cars):
    print('Saving inversion images')
    inversions_directory_path = os.path.join(args.save_dir, 'inversions')
    os.makedirs(inversions_directory_path, exist_ok=True)
    for i in tqdm(range(args.n_sample)):
        imgs, _ = g([latent_codes[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
        save_image(imgs[0], inversions_directory_path, img_names[i])


def run_alignment(image_path):
    predictor = dlib.shape_predictor(paths_config.model_paths['shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default='/home/vimlab/Downloads/face_resize_eyediap/',
                        help="The directory of the images to be inverted")
    parser.add_argument("--save_dir", type=str, default='./infer_resized_eyediap_with_eth_encoder/',
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--latents_only", action="store_true", help="infer only the latent codes of the directory")
    parser.add_argument("--align", action="store_true", help="align face images before inference")
    parser.add_argument("ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")

    args = parser.parse_args()
    main(args)