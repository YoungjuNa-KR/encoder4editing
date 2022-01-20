import torch
from models.stylegan2.model import Generator



model = Generator(256, 512, 2, channel_multiplier=2)  #512 -> 256으로 변경함


saved = torch.load('./pretrained_models/face_512_3000kimg.pt')
# saved = torch.load('./pretrained_models/network-stapshot-018200.pt')

# print(type(saved))

# # print(saved)


# for name, weight in saved['g_ema'].items():
#     print(name, ":", weight.shape)



for name, value in model.named_parameters():
    print(name,":", value.detach().numpy().shape) 