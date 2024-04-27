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

def find_anomaly(json_file, img_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    entries_to_delete = [] 
    
    for idx, entry in enumerate(data):
        if len(entry.get('kp-1', [])) != 7:
            # Image with insufficient keypoints
            print(f"{entry['img']}: {len(entry.get('kp-1', []))} keypoints")
            image_path = os.path.join(img_dir, entry['img'])
            print(f"Deleting {entry['img']}")
            os.remove(image_path)
            entries_to_delete.append(idx)

    # Delete problematic entries from data
    for idx in reversed(entries_to_delete):
        del data[idx]

    # Write updated data back to the JSON file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

    print("Deleted images tha don't have 7 keypoints,\
           and removed them from the annotations.")

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
    find_anomaly('ann.json', 'images')



if __name__ == '__main__':
    main()