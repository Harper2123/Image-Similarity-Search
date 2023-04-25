import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import sql_helper
import matplotlib.pyplot as plt
import vector_database
from PIL import Image
from feature_extract import ResNetFeatureExtractor
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

def main():
    print("Welcome to the Application.")
    print("This application allows you to search for similar images.")
    print("Do you want to use our database of images to search for similar images?")
    choice = input("Y/N: ")
    
    if choice == "N" or choice == "n":
        vector_database.get_connection()
        collection_name = input("Enter the name of the collection you want to create: ")
        vector_database.drop_collection(collection_name=collection_name)
        vector_database.get_collection(collection_name=collection_name)
        print("Collection created.")
        print("Now we will transform the images to embeddings. We will be using ResNet50 as our feature extraction model.")
        feature_extractor = ResNetFeatureExtractor()
        image_dataset_location = input("Please provide the path to the folder where the image dataset lies: ")
        features_dict, _ = feature_extractor.extract_features_from_directory(image_dataset_location)
        print("We have successfully performed feature extraction.")
        print("Now we will create an index for the collection.")
        vector_database.create_index(collection_name=collection_name)
        print("Index successfully created.")
        print("Now we will insert the embeddings into the collection.")
        vector_database.insert_into_collection(collection_name=collection_name, features_dict=features_dict)
        print("Embeddings successfully inserted.")
        print("Now we will search for similar images.")
        query_image_location = input("Please provide the path to the folder containing image you want to search for: ")
        _, query_list = feature_extractor.extract_features_from_directory(query_image_location)
        top_k = int(input("Please provide the number of similar images you require: "))
        result_id = vector_database.search_vectors(collection_name=collection_name, query_vector=query_list, top_k=top_k)
        print(f"We have successfully retrieved Milvus IDs of {top_k} similar images")
        print("Retrieving File Names......")
        file_list = sql_helper.retrieve_results(feat_list=result_id)
        print("File Names retrieved successfully.")
        print(f"Here are {top_k} similar images...")
        vector_database.display_images(folder_path=image_dataset_location, file_names=file_list)

    elif choice == "Y" or choice == "y":
        vector_database.get_connection()
        collection_name = 'test3'
        vector_database.get_collection(collection_name=collection_name)
        image_dataset_location = 'images/feat-test'
        print("Now we will search for similar images.")
        feature_extractor = ResNetFeatureExtractor()
        query_image_location = input("Please provide the path to the folder containing image you want to search for: ")
        _, query_list = feature_extractor.extract_features_from_directory(query_image_location)
        top_k = int(input("Please provide the number of similar images you require: "))
        result_id = vector_database.search_vectors(collection_name=collection_name, query_vector=query_list, top_k=top_k)
        print(f"We have successfully retrieved Milvus IDs of {top_k} similar images")
        print("Retrieving File Names......")
        file_list = sql_helper.retrieve_results(feat_list=result_id)
        print("File Names retrieved successfully.")
        print(f"Here are {top_k} similar images...")
        vector_database.display_images(folder_path=image_dataset_location, file_names=file_list)

    else:
        print("Invalid Input. Please try again.")




if __name__ == '__main__':
    main()