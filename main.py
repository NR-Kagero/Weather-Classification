import numpy as np
from Preprocessing import *
from torchvision import models
from torchsummary import summary

resNet50 = models.resnet50(weights='IMAGENET1K_V2')
for param in resNet50.parameters():
    param.requires_grad = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loaded_model = resNet50.to(device)

summary(resNet50, (3, 250, 250), 64, device=device)

X_train, Y_train = load_all("C:\\Users\\Kagero\\PycharmProjects\\weather_classification\\nature-dataset\\train")
data_loader_train = get_dataloader(X_train, Y_train, batch_size=64, shuffle=False)

extracted_features = feature_extraction(data_loader_train, loaded_model, device)
extracted_features = np.array(extracted_features)
labels = np.array(Y_train)
np.save("Data/extracted_features_train.npy", extracted_features)
np.save("Data/labels_train.npy", labels)

X_test, Y_test = load_all("C:\\Users\\Kagero\\PycharmProjects\\weather_classification\\nature-dataset\\test")
data_loader_test = get_dataloader(X_test, Y_test, batch_size=64, shuffle=False)
extracted_features_test = feature_extraction(data_loader_test, loaded_model, device)
labels_test = np.array(Y_test)
np.save("Data/extracted_features_test.npy", extracted_features_test)
np.save("Data/labels_test.npy", labels_test)