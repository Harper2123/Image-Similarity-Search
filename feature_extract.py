from keras.applications.resnet import ResNet50
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
from collections import OrderedDict
import numpy as np
import os
import sklearn.preprocessing

class ResNetFeatureExtractor:
    """
    Class for extracting ResNet50 features from images using Keras.
    """

    def __init__(self):
        """
        Initializes a ResNet50 model with pre-trained ImageNet weights.
        """
        self.model = ResNet50(weights='imagenet')

    def extract_features(self, img_path):
        """
        Extracts ResNet50 features from an image file.

        Parameters:
            img_path (str): Path to the input image file.

        Returns:
            features (numpy.ndarray): A 1D array of ResNet50 features for the input image.
        """
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = self.model.predict(x)

        return features.flatten()

    def extract_features_from_directory(self, directory):
        """
        Extracts ResNet50 features from all images in a directory.

        Parameters:
            directory (str): Path to the directory containing the input images.

        Returns:
            features_dict (collections.OrderedDict): A dictionary mapping image file names to their
                                                     ResNet50 features.
            feature_list (list): A list of ResNet50 features for all images in the directory.
        """
        features_dict = OrderedDict()
        feature_list = []
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                img_path = os.path.join(directory, filename)
                features = self.extract_features(img_path)
                #Normalizing
                features = sklearn.preprocessing.normalize(features.reshape(1,-1), axis=1, norm='l1')
                features_dict[filename] = features
                feature_list.extend(features)

        return features_dict, feature_list


def main():
    f1 = ResNetFeatureExtractor()
    loc = input("Enter the location of the image: ")
    feat_dict, feat_list = f1.extract_features_from_directory(loc)
    return feat_dict, feat_list

if __name__ == '__main__':
    feat_dict, feat_list = main()