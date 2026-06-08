# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import cv2
import os
from PIL import Image
import shutil

from torch.cuda.amp import autocast

from canvas_1024 import NeuralCanvas, NeuralCanvasStitched
from transforms import RandomRotate, Normalization, RandomCrop, RandomScale
from viz import *
from SDCGAN_sgmd import *

# map_location=torch.device('cpu')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# all 0 to 1
ACTIONS_TO_IDX = {
    'pressure': 0,
    'size': 1,
    'control_x': 2,
    'control_y': 3,
    'end_x': 4,
    'end_y': 5,
    'color_r': 6,
    'color_g': 7,
    'color_b': 8,
    'start_x': 9,
    'start_y': 10,
    'entry_pressure': 11,
    'pressure2': 12,
    'size2': 13,
    'control_x2': 14,
    'control_y2': 15,
    'end_x2': 16,
    'end_y2': 17,
    'color_r2': 18,
    'color_g2': 19,
    'color_b2': 20,
    'start_x2': 21,
    'start_y2': 22,
    'entry_pressure2': 23,
    'entry_pressure3': 24,
    'pressure3': 25,
    'size3': 26,
    'control_x3': 27,
    'control_y3': 28,
    'end_x3': 29,
    'end_y3': 30,
    'color_r3': 31,
    'color_g3': 32,
    'color_b3': 33,
    'start_x3': 34,
    'start_y3': 35,
    'entry_pressure33': 36,
    'pressure23': 37,
    'size23': 38,
    'control_x23': 39,
    'control_y23': 40,
    'end_x23': 41,
    'end_y23': 42,
    'color_r23': 43,
    'color_g23': 44,
    'color_b23': 45,
    'start_x23': 46,
    'start_y23': 47,
    'entry_pressure23': 48,
    'entry_pressure333': 49,
}


def pad(img, H, W):
    b, c, h, w = img.shape
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = torch.cat([torch.zeros((b, c, pad_h, w), device=img.device), img,
                     torch.zeros((b, c, pad_h + remainder_h, w), device=img.device)], dim=-2)
    img = torch.cat([torch.zeros((b, c, H, pad_w), device=img.device), img,
                     torch.zeros((b, c, H, pad_w + remainder_w), device=img.device)], dim=-1)
    return img


inception_v1 = torch.hub.load('pytorch/vision:v0.9.0', 'googlenet', pretrained=True)
resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
# vgg19 = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)

STROKES_PER_BLOCK = 1  # @param {type:"slider", min:1, max:15, step:1}

LAYER = "3B"  # @param ["3A", "3B"]
LAYER_IDX = -12 if LAYER == "3A" else -13
# @markdown Which GoogleNet layer to use for content loss. Deeper layers (3B) result in more abstract results
STOCHASTIC = False  # @param {type:"boolean"}
# @markdown Experimental. Adding uncertainty may (or may not) help produce more robust images.
NORMALIZE = True  # @paAram {type:"boolean"}
LEARNING_RATE = 0.099  # @param {type: "number"}
img_path = 'img1024'  # @param {type: "string"}
# img_path = os.listdir(img_path)
mask_path = 'msk1024'
edge_path = 'eg1024'
output_path = 'ai_p_results'

for img_name in os.listdir(img_path):
    file_name = img_name  # the original image
    file_name = file_name.split('.')[0]  # the name of the image
    output_path_new = output_path + "/" + file_name
    if not os.path.exists(output_path_new):
        os.makedirs(output_path_new)

    img = img_path + "/" + file_name + '.png'
    mask = mask_path + "/" + file_name + '.png'
    edge = edge_path + "/" + file_name + '.png'

    # file_name = os.path.basename(IMAGE_NAME)
    # file_name = file_name.split('.')[0]
    # output_path =output_path+"/"+file_name
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    print('STROKES_PER_BLOCK: {}'.format(STROKES_PER_BLOCK))
    print('LAYER: {}'.format(LAYER))
    print('LAYER_IDX: {}'.format(LAYER_IDX))
    print('STOCHASTIC: {}'.format(STOCHASTIC))
    print('NORMALIZE: {}'.format(NORMALIZE))
    print('LEARNING RATE: {}'.format(LEARNING_RATE))
    print('IMAGE NAME: {}'.format(img))

    neural_painter = Generator(len(ACTIONS_TO_IDX), 64, 3).to(device)
    neural_painter.load_state_dict(torch.load('sgan/sdcgan50_4b2_fc_10.tar'))  # gpu
    # neural_painter.load_state_dict(torch.load('sgan/oil_painting.tar',map_location='cpu'))

    # Normalization expected by GoogleNet (images scaled to (-1, 1))
    normalizer = Normalization(torch.tensor([0.5, 0.5, 0.5]).to(device),
                               torch.tensor([0.5, 0.5, 0.5]).to(device))

    # Define image augmentations
    padder = nn.ConstantPad2d(12, 0.5)
    rand_crop_8 = RandomCrop(8)
    rand_scale = RandomScale([1 + (i - 5) / 50. for i in range(11)])
    random_rotater = RandomRotate(angle=5, same_throughout_batch=True)
    rand_crop_4 = RandomCrop(4)

    # Content layer
    # feature_extractor_vgg = nn.Sequential(*list(vgg19.children())[:1])
    # feature_extractor_vgg.eval().to(device)
    feature_extractor = nn.Sequential(*list(inception_v1.children())[:LAYER_IDX])
    feature_extractor.eval().to(device)
    feature_extractor_res = nn.Sequential(*list(resnet18.children())[:4])
    feature_extractor_res.eval().to(device)

    # Define canvas and action preprocessor
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    action_preprocessor = torch.sigmoid  # torch.sigmoid is the default action preprocessor
    # action_preprocessor = nn.Sequential(*list(resnet18.children())[:9], nn.Linear(in_features=1024, out_features=560, bias=True))
    canvas = []
    for i in range(0, 6):
        canvas.append(NeuralCanvasStitched(neural_painter=neural_painter, overlap_px=32,
                                           repeat_h=2 ** (i + 1) - 1, repeat_w=2 ** (i + 1) - 1,
                                           strokes_per_block=STROKES_PER_BLOCK,
                                           action_preprocessor=action_preprocessor))

    # Load input image
    o_image = Image.open(img)

    loader = transforms.Compose([
        # transforms.Resize([canvas.final_canvas_h, canvas.final_canvas_w]),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    image_o = loader(o_image).unsqueeze(0)[:, :3, :, :].to(device, torch.float)
    image_o = pad(image_o, 1024, 1024)
    # torchvision.utils.save_image(image_o, 'results/new_canvases_tp2_200/SagradaFamília1.png')
    # Load input image mask
    mask = Image.open(mask)
    loader = transforms.Compose([
        # transforms.Resize([canvas.final_canvas_h, canvas.final_canvas_w]),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    mask = loader(mask).unsqueeze(0)[:, :3, :, :].to(device, torch.float)
    mask = pad(mask, 1024, 1024)
    mask_bg = 1 - mask
    # torchvision.utils.save_image(mask, 'results/new_canvases_tp2_200/SagradaFamília1_mask.png')
    # torchvision.utils.save_image(mask_bg, 'cad_results/mask_bg/cad43_mask_bg.png')

    edge = Image.open(edge)
    print(edge)
    loader = transforms.Compose([
        # transforms.Resize([canvas.final_canvas_h, canvas.final_canvas_w]),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    edge = loader(edge).unsqueeze(0)[:, :3, :, :].to(device, torch.float)
    print(edge.shape)
    edge = pad(edge, 1024, 1024)

    # image = torch.cat((image, image, image), 1)
    output_canvas_o = torch.ones(1, 3, 1024, 1024).to(device)
    # torchvision.utils.save_image(output_canvas_o, 'temp/output_canvas_o' + '.png')
    # output_canvas = output_canvas_o*edge
    output_canvas = output_canvas_o
    # torchvision.utils.save_image(output_canvas, 'temp/output_canvas' +'.png')
    print("output_canvas_shape:", output_canvas.shape)
    print("output_canvas_type:", output_canvas.type)
    # output_canvas = output_canvas_o
    image = image_o * mask
    image_bg = image_o * mask_bg

    # loss_fn = torch.nn.SmoothL1Loss()
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.L1Loss()

    # f = open('./loss/loss.txt', 'w')
    n_pt = 500  # number of painting times
    intermediate_canvases = []
    intermediate_paint_fg = []
    intermediate_paint_bg = []
    for k in range(1, 5):
        output_canvas = F.interpolate(output_canvas, (64 * (2 ** k), 64 * (2 ** k)))
        temp_canvas = output_canvas
        image = F.interpolate(image_o, (64 * (2 ** k), 64 * (2 ** k)))
        actions = torch.FloatTensor(canvas[k].total_num_strokes, 1, len(ACTIONS_TO_IDX)).uniform_().to(device)
        optimizer = optim.Adam([actions.requires_grad_()], lr=LEARNING_RATE)
        for idx in range(n_pt + 1):
            optimizer.zero_grad()
            if idx == n_pt:
                output_canvas, intermediate_canvase = canvas[k](actions, temp_canvas.detach(), True)
            else:
                output_canvas = canvas[k](actions, temp_canvas.detach())[0]
            # Everything else below is for calculating the loss function for painting style
            stacked_canvas = torch.cat([output_canvas, image])
            augmented_canvas = stacked_canvas
            # Pass through pretrained
            output_features = feature_extractor(augmented_canvas)
            output_features_res = feature_extractor_res(augmented_canvas)

            cost = loss_fn(output_features[0], output_features[1]) * 0.05 + loss_fn(output_features_res[0],
                                                                                    output_features_res[
                                                                                        1]) * 0.95  # ResNet-oil
            #         cost = loss_fn(output_features[0], output_features[1]) * 0.5 + loss_fn(output_features_res[0], output_features_res[1]) * 0.5 # ResNet+gg
            #         f.write(str(idx)+"\t"+str(cost.item())+"\t"+str(cost2.item())+"\n")
            cost.backward()
            optimizer.step()
            if idx % 1 == 0:
                print(f'k {k}\tStep {idx}\tCost {cost.item()}')
                # torchvision.utils.save_image(output_canvas, output_path_new + "/"+ file_name+str(k) + '_' + str(idx) + '.png')
        intermediate_canvases.extend(intermediate_canvase)

    n = len(intermediate_canvases)
    # for i in range(n):
    # #     print(intermediate_canvases.shape)
    # #     intermediate_canvases[i] = intermediate_canvases[i][w_t:512-w_t,h_t:512-h_t]

    #     torchvision.utils.save_image(intermediate_canvases[idx], 'temp/intermediate_canvases' + str(idx) + '.png')
    # print(n)
    intermediate_paint_fg = [[] for _ in range(n)]
    intermediate_paint_bg = [[] for _ in range(n)]

    edge = edge * mask + mask_bg
    mask = mask.cpu()
    mask_bg = mask_bg.cpu()
    edge = edge.cpu()
    for idx in range(n):
        intermediate_paint_fg[idx] = intermediate_canvases[idx] * mask
        intermediate_paint_fg[idx] = intermediate_paint_fg[idx] + mask_bg
        intermediate_paint_bg[idx] = intermediate_canvases[idx] * mask_bg
        intermediate_paint_bg[idx] = intermediate_paint_bg[idx] + intermediate_canvases[n - 1] * mask

    intermediate_paint_fg.extend(intermediate_paint_bg)
    m = len(intermediate_paint_fg)
    paint_step = [[] for _ in range(m)]
    for i in range(m):
        paint_step[i] = intermediate_paint_fg[i] * edge


    def con(dir_image1, dir_image2):
        image1 = np.array(dir_image1)
        image2 = np.array(dir_image2)
        if (np.array_equal(image1, image2)):
            result = "con_same"
        else:
            result = "con_diff"
        return result


    def remove_list(lista, listb):
        for x in listb:
            lista.remove(x)
        return lista


    def con_com(dir_image1, dir_image2):

        # 比较两张图片是否相同
        result = "con_diff"
        re = con(dir_image1, dir_image2)
        if (re == "con_same"):
            result = "con_same"
        return result


    file_repeat = []
    for currIndex, filename in enumerate(paint_step):
        dir_image1 = paint_step[currIndex]
        dir_image2 = paint_step[currIndex + 1]
        result = con_com(dir_image1, dir_image2)
        if (result == "con_same"):
            file_repeat.append(currIndex + 1)
        #         print("\n相同的图片：", paint_step[currIndex], paint_step[currIndex + 1])
        #         del paint_step[currIndex+1]
        #     else:
        #         print('\n不同的图片：', intermediate_paint_fg[currIndex], intermediate_paint_fg[currIndex + 1])
        currIndex += 1
        if currIndex >= len(paint_step) - 1:
            break

    print(len(paint_step))
    print(len(file_repeat))

    file_repeat = file_repeat[::-1]
    for img in file_repeat:
        del paint_step[img]
    print(len(paint_step))
    for i in range(len(paint_step)):
        torchvision.utils.save_image(paint_step[i], output_path_new + "/" + file_name + '_' + str(i) + '.png')

    animate_strokes_on_canvas(paint_step, image_o, output_path_new + "/" + file_name + '.mp4', skip_every_n=1)
