from torchvision.models import inception_v3
from torchvision import transforms as T
import torch


IMAGENET_CLASSES_PATH='imagenet_classes.txt'

model = inception_v3(pretrained=True)
model.eval()

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open(IMAGENET_CLASSES_PATH) as f:
    labels = [line.strip().split()[1] for line in f.readlines()]


def classify(img):
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    output = model(batch_t)
    _, index = torch.max(output, dim=1)
    index = index.item()

    percentage = torch.softmax(output, dim=1)[0] * 100
    pred_label = labels[index]
    pred_confidence = percentage[index].item()
    return pred_label, pred_confidence

