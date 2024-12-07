import torch
import matplotlib.pyplot as plt
import cv2,random
import numpy as np
import sys
sys.path.append("C:\\Users\\Ribhav\\Desktop\\everything\\FaceLandmarks")
from core.models import FAN

model = FAN()
#model load
checkpoint = torch.load("C:\\Users\\Ribhav\\Desktop\\everything\\FaceLandmarks\\ckpt\\WFLW_4HG.pth", map_location='cpu')
#model = torch.load('./ckpt/WFLW_4HG.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

#load image


path = "C:\\Users\\Ribhav\\Desktop\\everything\\python_scripts\\Ocelot1_images\\outerBrow_up_right.jpg"
img = plt.imread(path)
img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)[:,:,:3]


plt.imshow(img)

#infer
input_ = torch.tensor(img).transpose(0,-1).view(1,3,256,256)
out = model(input_)[-1].squeeze().detach()
out = out.numpy()

ind = np.array([ [np.argmax(out[i])%64,np.argmax(out[i])//64] for i in range(0,68)]).astype(np.float64)

ind2 = list() #second largest index
mask = [[-1,-1],[-1,0],[-1,1],[1,0],[0,0],[0,1],[1,-1],[1,0],[1,1]]
for i in range(0,68):
    mx,my = int(ind[i,0]), int(ind[i,1])
    out[i,my,mx] -= 10
    neigbormax = np.argmax(out[i, my-1: my+2, mx-1:mx+2])
    out[i,my,mx] += 10
    dy, dx = mask[neigbormax]
    ind2.append([mx+dx, my+dy])

ind2 = np.array(ind2).astype(np.float64)
new = (ind*3+ind2)/4.

ind *= (256/64.)
ind2 *= (256/64.)
new *= (256/64.)

plt.scatter(new[:,0],new[:,1],c='black',s=10)

plt.show()