# 이하를 「model.py」에 써넣기
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

classes_kr = ["티셔츠/톱", "바지", "풀오버", "드레스", "코드", "샌들", "와이셔츠", "스니커즈", "가방", "앵클부츠"]
classes_en = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
n_class = len(classes_kr)
img_size = 28

# 이미지 인식의 모델
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv２ = nn.Conv2d(8, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64*4*4, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn2(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

net = Net()

# 훈련한 파라미터 읽어들이기와 설정
net.load_state_dict(torch.load(
    "model_cnn.pth", map_location=torch.device("cpu")
    ))
    
def predict(img):
    # 모델로의 입력
    img = img.convert("L")  # 흑백으로 변환
    img = img.resize((img_size, img_size))  # 크기를 변환
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.0), (1.0))
                                    ])
    img = transform(img)
    x = img.reshape(1, 1, img_size, img_size)
    
    # 예측
    net.eval()
    y = net(x)

    # 결과를 반환한다
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # 확률로 나타낸다
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 내림차순으로 정렬
    return [(classes_kr[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
