#@title Setup Repository
import os
from argparse import Namespace
import time
import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from utils.common import tensor2im
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.

class LatentEditor(object):
    def __init__(self, stylegan_generator):
        self.generator = stylegan_generator

    # Currently, in order to apply StyleFlow editings, one should run inference,
    # save the latent codes and load them form the official StyleFlow repository.
    # def apply_styleflow(self):
    #     pass

    def _latents_to_image(self, latents):
        with torch.no_grad():
            images, _ = self.generator([latents], randomize_noise=False, input_is_latent=True)
        horizontal_concat_image = torch.cat(list(images), 2)
        final_image = tensor2im(horizontal_concat_image)
        return final_image
experiment_type = 'ffhq_encode'
EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "pretrained_models/eth_eht_iteration_200000.pt",
    }
}
# Setup required image transformations
EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

EXPERIMENT_ARGS['transform'] = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

resize_dims = (256, 256)

model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')
# using opts from ckpt
opts = ckpt['opts']
# pprint.pprint(opts)  # Display full options used
# update the training options. add checkpoint_path key and value
opts['checkpoint_path'] = model_path
opts= Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')
editor = LatentEditor(net.decoder)

# mpii image
image_path_mp = '/home/vimlab/Downloads/lab/mpii_test/p00_2277.jpg'
img_name_mp = image_path_mp.split('/')[-1].split('.')[0]
# eth image
image_path_eth = '/home/vimlab/Downloads/eth_256_edit/val/0000_620.jpg'
img_name_eth = image_path_eth.split('/')[-1].split('.')[0]


# mpii
original_image_mp = Image.open(image_path_mp)
original_image_mp = original_image_mp.convert("RGB")

#eth
original_image_eth = Image.open(image_path_eth)
original_image_eth = original_image_eth.convert("RGB")

img_transforms = EXPERIMENT_ARGS['transform']
transformed_image_mp = img_transforms(original_image_mp)
transformed_image_eth = img_transforms(original_image_eth)


def display_alongside_source_image(result_image, source_image):
    res = np.concatenate([np.array(result_image.resize(resize_dims)),
                          np.array(source_image.resize(resize_dims))], axis=1)
    result_image = Image.fromarray(res)
    return result_image

def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    if experiment_type == 'cars_encode':
        images = images[:, :, 32:224, :]
    return images, latents

with torch.no_grad():
    tic = time.time()
    # PIL -> tensor -> runonbatch
    images_mp, latents_mp = run_on_batch(transformed_image_mp.unsqueeze(0), net)
    images_eth, latents_eth = run_on_batch(transformed_image_eth.unsqueeze(0), net)
    result_image_mp, latent_mp = images_mp[0], latents_mp[0]
    result_image_eth, latent_eth = images_eth[0], latents_eth[0]
    
    toc = time.time()
    print('Inference took {:.4f} seconds.'.format(toc - tic))
    # print(torch.equal(result_image_latent[0], latent))
    print(latent_eth.shape)


# pt 저장
torch.save(latent_mp, f'./{img_name_mp}.pt')
torch.save(latent_eth, f'./{img_name_eth}.pt')

before = latent_eth.clone().detach()

latent_eth[5:9][:] = latent_mp[5:9][:]

print(before == latent_eth)


print(latent_eth.shape)
save_img = editor._latents_to_image(latent_eth.unsqueeze(0))

save_img.save('./mixed_latent.jpg')