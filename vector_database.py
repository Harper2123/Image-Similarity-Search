import os
import numpy as np
import sql_helper
import matplotlib.pyplot as plt
from PIL import Image
from feature_extract import ResNetFeatureExtractor

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


# Creating Connection
def get_connection(alias='default', host='localhost', port='19530'):
    """
    Creates a connection to the Milvus server.

    Parameters
    ----------
    name : str
        The name of the connection.
    host : str
        The host address of the Milvus server.
    port : str
        The port number of the Milvus server.
    """
    try:
        connections.connect(alias=alias, host=host, port=port)
        print("Connection successfully established")
    except Exception as e:
        print("Error while connecting to Milvus: ", e)

# Creating Collection
def get_collection(collection_name, host='localhost', port='19530'):
    """
    Gets or creates the collection with the specified name.

    Returns
    -------
    Collection
        The collection object.
    """
    # Checking if Collection already exists
    if collection_name in utility.list_collections():
        #print("The collection {" + collection_name + "} already exists")
        return Collection(collection_name)
    else:
        #print("The collection {" + collection_name + "} does not exist")
        # Creating the collection
        #print("Creating the collection")
        id_field = FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True, description='Primary Milvus id')
        feature_field = FieldSchema(name='Embedding', dtype=DataType.FLOAT_VECTOR, dim=1000, is_primary=False, description='Normalized Feature Vectors')
        schema = CollectionSchema(fields=[id_field, feature_field], description='Collection of Feature Vectors')
        
        collection1 = Collection(name=collection_name, schema=schema)
        return collection1

# Dropping Collection
def drop_collection(collection_name, host='localhost', port='19530'):
    """
    Drops the collection with the specified name.

    Parameters
    ----------
    collection_name : str
        The name of the collection to drop.
    """
    try:
        collection = get_collection(collection_name)
        collection.drop()
        print("Collection {" + collection_name + "} dropped")
    except Exception as e:
        print("Error while dropping collection: ", e)

# Creating Index
def create_index(collection_name):
    """
    Creates an index for the specified collection.

    Parameters
    ----------
    collection_name : str
        The name of the collection to create an index for.
    """

    #Defining Index Parameters
    index_params = {
        "metric_type":"IP",
        "index_type":"IVF_FLAT",
        "params":{
            "nlist": 500
        }
    }
    try:
        collection = get_collection(collection_name)
        collection.create_index(field_name='Embedding', index_params=index_params)
        utility.index_building_progress(collection_name)
        print("Index created for collection {" + collection_name + "}")
    except Exception as e:
        print("Error while creating index: ", e)


# Inserting Vectors
def insert_into_collection(collection_name, features_dict):
    try:
        collection = get_collection(collection_name=collection_name)
        for filename, feature_vector in features_dict.items():
            #Inserting a single vector
            data = [np.array(feature_vector, dtype=np.float32)]
            ins = collection.insert(data)
            milvus_id = ins.primary_keys[0]
            #Inserting the filenames and milvus id into SQL
            sql_helper.insert_into_sql(filename, milvus_id)
        collection.load()
        print("Vectors successfully inserted")
    except Exception as e:
        print("Error while inserting vectors: ", e)

# Searching Vectors
def search_vectors(collection_name, query_vector, top_k=50):
    """
    Searches the specified collection for the specified vector.

    Parameters
    ----------
    collection_name : str
        The name of the collection to search.
    query_vector : list
        The vector to search for.
    top_k : int
        The number of top results to return.

    Returns
    -------
    list
        The top results.
    """

    search_params = {
        "metric_type":"IP",
        "params":{"nprobe": 300}
    }
    try:
        collection = get_collection(collection_name)
        query = np.array(query_vector, dtype=np.float32)
        results = collection.search(query, "Embedding", param=search_params, limit=top_k)
        #print(results)
        return results[0].ids
    except Exception as e:
        print("Error while searching vectors: ", e)

def get_images(file_names,folder_path):
    """
    Returns the images from the specified folder.

    Parameters
    ----------
    file_names : list
        The list of file names to get.
    folder_path : str
        The path to the folder to get the images from.

    Returns
    -------
    list
        The list of images.
    """
    images = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = Image.open(file_path)
        images.append(image)
    return images

def display_images(folder_path, file_names):
    """
    Displays the images from the specified folder.

    Parameters
    ----------
    file_names : list
        The list of file names to display.
    folder_path : str
        The path to the folder to display the images from.
    """
    images = get_images(file_names, folder_path)
    for image in images:
        plt.imshow(image)
        plt.show()

def main():
    # Creating Connection
    get_connection()

    # Creating Collection
    collection_name = 'test3'
    drop_collection(collection_name)
    get_collection(collection_name)

    # Extracting Features
    feature_extractor = ResNetFeatureExtractor()
    features_dict, _ = feature_extractor.extract_features_from_directory('images/feat-test')
    _, test_list = feature_extractor.extract_features_from_directory('images/search-test')
    # Creating Index
    create_index(collection_name)

    # Inserting Vectors
    insert_into_collection(collection_name, features_dict)

    # Searching Vectors
    result_id = search_vectors(collection_name=collection_name,query_vector=test_list)

    # Retrieving file names of similar images
    file_list = sql_helper.retrieve_results(feat_list=result_id)

    # Retrieving images
    folder = 'images/feat-test'
    display_images(folder, file_list)

if __name__ == '__main__':
    main()
    
