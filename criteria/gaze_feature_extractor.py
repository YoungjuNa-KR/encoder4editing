import torch
import torch.nn as nn
from torchvision import models


class GazeEstimationAbstractModel(nn.Module):

    def __init__(self):
        super(GazeEstimationAbstractModel, self).__init__()

    @staticmethod
    def _create_fc_layers(in_features, out_features):

        output_layer = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features)
        )

        return  output_layer

    def forward(self, face_img):

        feature = self.feature_extractor(face_img)
        feature = feature.view(feature.shape[0], -1)
        reduction = self.reduction(feature)
        # output = self.fc(reduction)

        return reduction
    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)


class GazeEstimationAbstractModelLatent(nn.Module):

    def __init__(self):
        super(GazeEstimationAbstractModelLatent, self).__init__()

    @staticmethod
    def _create_fc_layers(in_features, out_features):

        output_layer = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features)
        )

        return  output_layer

    def forward(self, latent_vector):
        latent_vector = latent_vector.view(latent_vector.shape[0], -1)
        output = self.fc(latent_vector)

        return output

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

class GazeEstimationAbstractModel_img_with_latent(nn.Module):

    def __init__(self):
        super(GazeEstimationAbstractModel_img_with_latent, self).__init__()

    @staticmethod
    def _create_fc_layers(in_features, out_features):

        output_layer = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features)
        )

        return  output_layer

    def forward(self, face_img, latent_vector):

        feature = self.feature_extractor(face_img)
        feature = feature.view(feature.shape[0], -1)
        
        latent =  latent_vector.view(latent_vector.shape[0], -1)

        feature = torch.cat((feature, latent), 1)


        reduction = self.reduction(feature)
        output = self.fc(reduction)

        return output

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)
                

class GazeEstimationModelResnet18(GazeEstimationAbstractModel):
    
    def __init__(self, num_out=2):
        super(GazeEstimationModelResnet18, self).__init__()
        feature_extractor = models.resnet18(pretrained=True)

        # remove the last ConvBRelu layer
        self.feature_extractor = nn.Sequential(
            feature_extractor.conv1,
            feature_extractor.bn1,
            feature_extractor.relu,
            feature_extractor.maxpool,
            feature_extractor.layer1,
            feature_extractor.layer2,
            feature_extractor.layer3,
            feature_extractor.layer4,
            feature_extractor.avgpool
        )
        
        self.reduction = nn.Linear(512, 2)

        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        

class GazeEstimationModelResnet101(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationModelResnet101, self).__init__()
        feature_extractor = models.resnet101(pretrained=True)

        self.feature_extractor = nn.Sequential(
            feature_extractor.conv1,
            feature_extractor.bn1,
            feature_extractor.relu,
            feature_extractor.maxpool,
            feature_extractor.layer1,
            feature_extractor.layer2,
            feature_extractor.layer3,
            feature_extractor.layer4,
            feature_extractor.avgpool
        )
        
        self.reduction = nn.Linear(2048, 2)

        for param in self.feature_extractor.parameters():
            param.requires_grad = True


        # self.fc = GazeEstimationAbstractModel._create_fc_layers(in_features=300, out_features=2)
        # GazeEstimationAbstractModel._init_weights(self.modules())

class GazeEstimationModelVGG16(GazeEstimationAbstractModel):
    def __init__(self, num_out=2):
        super(GazeEstimationModelVGG16, self).__init__()
        feature_extractor = models.vgg16(pretrained=True)

        feature_extractor_modules = [module for module in feature_extractor.features]
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        feature_extractor_modules.append(self.AdaptiveAvgPool2d)
        self.feature_extractor = nn.Sequential(*feature_extractor_modules)

        # self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.reduction = nn.Linear(512, 300)

        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.fc = GazeEstimationAbstractModel._create_fc_layers(in_features=300, out_features=2)
        GazeEstimationAbstractModel._init_weights(self.modules())

class GazeEstimationModelLatent(GazeEstimationAbstractModelLatent):
    def __init__(self, num_out=2):
        super(GazeEstimationModelLatent, self).__init__()
        self.fc = GazeEstimationAbstractModelLatent._create_fc_layers(in_features=300, out_features=2)
        GazeEstimationAbstractModelLatent._init_weights(self.modules())


class GazeEstimationModel_img_with_Latent(GazeEstimationAbstractModel_img_with_latent):
    def __init__(self):
        super(GazeEstimationModel_img_with_Latent, self).__init__()
        feature_extractor = models.resnet101(pretrained=True)

        self.feature_extractor = nn.Sequential(
            feature_extractor.conv1,
            feature_extractor.bn1,
            feature_extractor.relu,
            feature_extractor.maxpool,
            feature_extractor.layer1,
            feature_extractor.layer2,
            feature_extractor.layer3,
            feature_extractor.layer4,
            feature_extractor.avgpool
        )

        self.reduction = nn.Linear(10240, 300)

        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.fc = GazeEstimationAbstractModel_img_with_latent._create_fc_layers(in_features=300, out_features=2)
        GazeEstimationAbstractModel_img_with_latent._init_weights(self.modules())

       
def make_model(model):
    if model == "ResNet18":
        return GazeEstimationModelResnet18()
    elif model == "VGG16":
        return GazeEstimationModelVGG16()
    if model == "Img_with_Latent":
        print("Img_with_Latent")
        return GazeEstimationModel_img_with_Latent()

class GazeModel(nn.Module):
    def __init__(self, opt):
        super(GazeModel, self).__init__()
        print('\nMaking Gaze model...')
        self.opt = opt
        self.n_GPUs = opt.n_GPUs
        self.device = torch.device('cpu' if opt.cpu else 'cuda')
        self.model = make_model(opt.model).to(self.device)

    def load(self, pre_train, cpu=False):
        
        #### load gaze model ####
        if pre_train != '.':
            print('Loading gaze model from {}'.format(pre_train))
            self.model.load_state_dict(
                torch.load(pre_train),
                strict=True
            )
            print("Complete loading Gaze estimation model weight")
        
        num_parameter = self.count_parameters(self.model)

        print(f"The number of parameters is {num_parameter / 1000 ** 2:.2f}M \n")

    def forward(self, imgs ):
        return self.model(imgs)

    def count_parameters(self, model):
        param_sum = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return param_sum