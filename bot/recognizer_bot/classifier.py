import torch
from torchvision.models import inception_v3
from torchvision import transforms as T


class Classifier():
    def __init__(self, classes_path):

        # Upload labels list for classifier output
        self.labels = self.upload_labels(classes_path)

        # Prepre pretrained model
        model = inception_v3(pretrained=True)
        model.eval()
        self.model = model

        # Set image transformation for inputs
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def upload_labels(classes_path):
        with open(classes_path) as f:
            labels = [line.strip().split()[1] for line in f.readlines()]
        return labels

    def classify(self, img):
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        output = self.model(batch_t)
        _, index = torch.max(output, dim=1)
        index = index.item()

        percentage = torch.softmax(output, dim=1)[0] * 100
        pred_label = self.labels[index]
        pred_confidence = percentage[index].item()
        return pred_label, pred_confidence
