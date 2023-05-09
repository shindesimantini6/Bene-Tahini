# preprocessing the images

#%%

import glob
import json
import os
import shutil

def add_element_and_get_index(element, element_list):
    """
    Checks if an element is in a list and adds it to the list if not.
    Returns the index of the element in the list.

    Args:
        element (str):
            Element to be added to the list
        element_list (list):
            List to add the element to

    Returns:
         Index of inserted element in the list
    """

    if element not in element_list:
        element_list.append(element)
    return element_list.index(element)


# Read the json files from annotations and get the chart type

# Empty list to store the class names
classes = []

# Paths to the train images and annotations
base_path_annotations = '/home/shinde/Documents/personal_projects/Bene-Tahini/benetech-making-graphs-accessible/train/annotations/*'
base_path_images  = '/home/shinde/Documents/personal_projects/Bene-Tahini/benetech-making-graphs-accessible/train/images/*'

# Loop through the json and image files to match the name and get the class name (chart type) from json file
for i in glob.glob(base_path_annotations):
    # print(i.split("/")[-1])
    # Opening JSON file
    f = open(i)
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    print(data["chart-type"])
    class_name = data["chart-type"]
    add_element_and_get_index(class_name, classes)
    
    dest_fpath = f"/home/shinde/Documents/personal_projects/Bene-Tahini/benetech-making-graphs-accessible/train_classes/{class_name}/"
    # print(dest_fpath)
    if not os.path.exists(dest_fpath):
        os.mkdir(os.path.dirname(dest_fpath))
    for j in glob.glob(base_path_images):
        # print(j.split("/")[-1].split(".")[0])
        if i.split("/")[-1].split(".")[0] == j.split("/")[-1].split(".")[0]:
            print("true")
            shutil.copy(j, dest_fpath)


# %%
