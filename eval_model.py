import torch
import numpy as np
from PIL import Image
from custom_fcn import custom_fcn
from torchvision import datasets, models, transforms 
import PIL
import os
import shutil


def load_custom_model():
    model = custom_fcn(n_class = 21)
    model_weights = torch.load('./weights/custom_model_best.pth')
    model.load_state_dict(model_weights)
    model = model.cuda()
    model.eval()
    return model

def load_fcn_model():
    model = torch.load('./weights/resnet50_weights.pt')
    model = model.cuda()
    model.eval()
    return model

def draw_segmentation_masks(mask, img_in):
    print("img size: ", img_in.shape)
    print("mask shape: ", mask.shape)
    img = img_in.copy()
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

    return img

def compare_models(img_list, seg_list):
    custom = load_custom_model()
    fcn_model = load_fcn_model()

    img_tforms = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    for id in range(0, len(img_list)):
        img = img_list[id]
        seg = seg_list[id]
        # print("img size: ", img.size)
        img_tensor = img_tforms(img).cuda()
        # print("img tensor shape: ", img_tensor.shape)
        
        # exit()
        a,b,c = img_tensor.shape
        img_tformed = img_tensor.reshape(1, a, b, c)    

        custom_preds = custom(img_tformed)
        fcn_preds = fcn_model(img_tformed)['out']

        custom_idxs = torch.argmax(custom_preds.squeeze(), dim=0).detach().cpu().numpy()
        fcn_idxs = torch.argmax(fcn_preds.squeeze(), dim=0).detach().cpu().numpy()

        # print("custom idxs: ", custom_preds[0, :, 0, 0])
        print("custom idxs unique:", np.unique(custom_idxs))

        img_np = np.array(img.copy())
        seg_np = np.array(seg.copy())

        # print("img np shape:" , img_np.shape)
        
        custom_seg_img = draw_segmentation_masks(custom_idxs, img_np)
        fcn_seg_img = draw_segmentation_masks(fcn_idxs, img_np) 
        gt_seg_img = draw_segmentation_masks(seg_np, img_np)

        res_img = np.hstack([img_np, custom_seg_img, fcn_seg_img, gt_seg_img]) 

        print("image stack shape: ", res_img.shape)
        
        res_img = PIL.Image.fromarray(np.uint8(res_img))
        res_img.save('./figures/results/' + str(id) + '_res.jpg')    


def edit_image(img):
    w,h = img.size
    new_h = 512 
    new_w = int(512 * w / h)
    overflow = new_w % 64
    
    print("orig img size: ", img.size)

    img_rs = img.resize((new_w, new_h))
    print("resized img size: ", img_rs.size)
    img_rs = img_rs.crop((0, 0, new_w - overflow, new_h))
    print("cropped img size: ", img_rs.size)

    # im1 = im.crop((left, top, right, bottom))

    return img_rs 

def sample_images():
    text_file = './data_pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/' + 'val' + '.txt'
    img_root_path = './data_pascal/VOCdevkit/VOC2012/' + 'JPEGImages'
    seg_root_path = './data_pascal/VOCdevkit/VOC2012/' + 'SegmentationClass'
    dst_path = './figures/test_set' 
    img_names = [] 

    with open(text_file, 'r') as fl:
        for name in fl:
            name_edited = name[:-1]
            img_names.append(name_edited) 

    for id in range(0, len(img_names)):
        orig_img = os.path.join(img_root_path, img_names[id] + '.jpg')
        seg_img = os.path.join(seg_root_path, img_names[id] + '.png')   

        img_save_path = os.path.join(dst_path, str(id) + '_orig_.jpg')
        seg_save_path = os.path.join(dst_path, str(id) + '_seg_.jpg')

        shutil.copy(orig_img, img_save_path)
        shutil.copy(seg_img, seg_save_path) 

def run_main():
    # img_name_list = ['./figures/dinner.jpeg', './figures/sofa2.jpeg', './figures/sofa.jpeg']
    img_name_list = ['./figures/test_set/' + str(id) + '_orig_.jpg' for id in range(0,20)]
    img_seg_list = ['./figures/test_set/' + str(id) + '_seg_.jpg' for id in range(0,20)]
    
    imgs = []
    segs = []
    for id in range(len(img_name_list)):
        img_path = img_name_list[id]
        seg_path = img_seg_list[id]

        img = Image.open(img_path)
        seg = Image.open(seg_path)
        # Skipping the image if its is portrait 
        if (img.size[1] > img.size[0]): 
            continue
        img_edit = edit_image(img)
        seg_edit = edit_image(seg)
        imgs.append(img_edit) 
        segs.append(seg_edit)

    compare_models(imgs, segs) 
    # sample_images()


if __name__ == "__main__":
    run_main()