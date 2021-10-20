
import vgg_torch
import resnet


def get_model_by_name(model_name):
    model = None
    if (model_name=='resnet18'):
        model = resnet.ResNet18()
    elif (model_name=='resnet34'):
        model = resnet.ResNet34()
    elif (model_name=='resnet50'):
        model = resnet.ResNet50()
    elif (model_name=='resnet101'):
        model = resnet.ResNet101()
    elif (model_name=='resnet152'):
        model = resnet.ResNet152()
    elif (model_name=='vgg11'):
        model = vgg_torch.vgg11_bn()
    elif (model_name=='vgg13'):
        model = vgg_torch.vgg13_bn()
    elif (model_name=='vgg16'):
        model = vgg_torch.vgg16_bn()
    elif (model_name=='vgg19'):
        model = vgg_torch.vgg19_bn()
    else:
        print('only ResNet and VGG models are supported for now!')
    return model