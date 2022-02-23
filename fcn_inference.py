import PIL
from PIL import Image 
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms

# This module is to visualize the semgnetation masks 
def visualize_segms(x):
    sem_classes = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    normalized_masks = x

    person_masks = [
        normalized_masks[img_idx, sem_class_to_idx[cls]]
        for img_idx in range(1)
        for cls in ('person', 'bus') 
    ]
    return person_masks

def draw_segmentation_masks(mask, img):
    color_pallete = [(0, 0, 0),  # 0=background
                    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]


    for id in range(0, len(color_pallete)):
        idxs = mask == id

        img[idxs, 0] = color_pallete[id][0]
        img[idxs, 1] = color_pallete[id][1]
        img[idxs, 2] = color_pallete[id][2] 


def forward_pass(img, model):
    img_tforms = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ]) 

    img_tformed = img_tforms(img).cuda()
    a,b,c = img_tformed.shape
    img_tformed = img_tformed.reshape(1, a, b, c)

    model_preds = model(img_tformed)['out']
    print("model preds shape: ", model_preds.shape)
    model_preds = torch.argmax(model_preds.squeeze(), dim=0).detach().cpu().numpy()
    return model_preds


# This is the main function that will load a pre-trained fcn model and set of input images and calls the inference. 
def run_main():
    # IT is very slow to downlaod the model, we can use the downloaded model weights for the inference task 
    # fcn_model = torchvision.models.segmentation.fcn_resnet101(pretrained = False) 
    fcn_model = torch.load('./weights/resnet50_weights.pt')
    fcn_model = fcn_model.cuda()

    inp_img = Image.open('./figures/dinner.jpeg')
    h,w = inp_img.size
    model_preds = forward_pass(inp_img, fcn_model)

    print("input image shape: ", inp_img.size)
    print("model preds shape: ", model_preds.shape)  

    out_img = np.array(inp_img.resize((int(h*512/w), 512)))
    draw_segmentation_masks(model_preds, out_img)

    out_img = PIL.Image.fromarray(np.uint8(out_img)) 
    out_img.save('./figures/inp3_fc_seg.png')  

if __name__ == "__main__":
    run_main()
    





