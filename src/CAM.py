# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

from dcnn import *
import io
import requests
import cv2

class Bbox_set(Dataset):
    """chest x-ray boboxe test dataset."""

    classes = {'Atelectasis':0, 'Cardiomegaly':1, 'Effusion':2, 'Infiltrate':3, \
                        'Mass':4, 'Nodule':5, 'Pneumonia':6, 'Pneumothorax':7, \
                        'Consolidation':8, 'Edema':9, 'Emphysema':10, 'Fibrosis':11, \
                        'Pleural_Thickening':12, 'Hernia':13   }
    
    def __init__(self, csv_bboxfile=join(path,'BBox_list_2017.csv'), \
                 root_dir=join(path,'images/images'), transform=None):
        """
        Args:
            csv_bboxfile (string): Path to the csv file with bbox.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.bbox=pd.read_csv(csv_bboxfile)             
        self.root_dir = root_dir        
        self.transform = transform

    def __len__(self):
        return len(self.bbox)

    def __getitem__(self, idx):
        img_name = self.bbox.iloc[idx, 0]
        image = Image.open(join(self.root_dir, img_name)).convert('RGB')
        label = self.classes[self.bbox.iloc[idx, 1]]
        
        bbox = self.bbox.iloc[idx,[2,3,4,5]].as_matrix()
        bbox = bbox.astype(np.float32)
        
        sample = {'image': image, 'label': label, 'name': img_name, 'bbox': bbox}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
checkpoint=torch.load('Alex_model_best.pth.tar')
net = MyAlexNet()
net.load_state_dict(checkpoint['state_dict'])
net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules['features']._modules['transit'].register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, size, class_idx):
    # generate the class activation maps upsample to size
    size_upsample = size if isinstance(size, tuple) else (size, size)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


bbox_set=Bbox_set(transform=transform)
bbox_loader = DataLoader(bbox_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
for i, data in enumerate(bbox_loader):
    img_variable, label, name, bbox=data['image'], data['label'].numpy()[0], data['name'][0], data['bbox'].numpy()[0]
    logit = net(Variable(img_variable, volatile=True))
    CAMs = returnCAM(features_blobs[0], weight_softmax, 1024, [label])
    features_blobs = []
    heatmap = cv2.applyColorMap(CAMs[0],cv2.COLORMAP_JET)
    cv2.rectangle(heatmap, tuple(bbox[:2]), tuple(bbox[:2]+bbox[2:4]), (255,0,0), 5)
    text='label:{}, pro:{:.3f}'.format(label,F.sigmoid(logit[:,label]).data.numpy()[0])
    cv2.putText(heatmap, text, (10, 1014),  cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5)
    cv2.imwrite(join(path,'heatmaps','label'+str(label)+'_'+name), heatmap)
        
    print("loop: {}/{}, image:{}, label:{}, probability: \
          {}".format(i,len(bbox_loader),name,label,F.sigmoid(logit[:,label]).data.numpy()[0]))

# generate class activation mapping for the top1 prediction


#cv2.rectangle(img, (212,317), (290,436), (0,255,0), 4)