import os
import json
import shutil

def copy_images(json_file, image_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract image names from JSON
    image_names = [entry['img'] for entry in data]
    
    total = 0    

    # Copy all images that are mentioned in ann.json
    # to output_folder    
    for image_name in image_names:
        src_path = os.path.join(image_folder, image_name)
        if os.path.exists(src_path):
            dest_path = os.path.join(output_folder, image_name)
            shutil.copyfile(src_path, dest_path)
            print(f"Image {src_path} copied successfully.")

        total += 1

    print(f"Total images in JSON: {total}")


def delete_duplicates(folder1_del, folder2_keep):
    """Deletes images that exist in both folders from folder1_del"""
    # Get the list of images in folder2_keep, these images already exist
    images_to_keep = os.listdir(folder2_keep)
    
    deleted = 0
    # Iterate over the images in folder1_del
    for image_name in os.listdir(folder1_del):
        # Check if the image already exists in folder2_keep
        if image_name in images_to_keep:
            # Delete the image from folder1_del
            image_path = os.path.join(folder1_del, image_name)
            os.remove(image_path)
            print(f"Deleted duplicate image: {image_name}")
            deleted += 1
    print(f"Deleted {deleted} duplicate images from {folder1_del}.")
    print(f"These images still exist inside {folder2_keep}")


def main():
    ## COPY_IMAGES
    # # Paths
    # json_file = 'ann.json'
    # image_folder = 'all_images'
    # output_folder = 'verified_images'

    # # Call the function
    # copy_images(json_file, image_folder, output_folder)

    ## DELETE_DUPLICATES
    # delete_duplicates("all_images", "verified_images")
    pass


if __name__ == '__main__':
    main()