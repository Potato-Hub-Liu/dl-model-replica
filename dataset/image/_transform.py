import torchvision.transforms as transforms

mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)])