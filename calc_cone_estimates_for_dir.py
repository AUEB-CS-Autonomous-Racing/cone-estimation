import os
from cone_estimation import cone_estimation
def calc_cone_estimates_for_dir():
    image_paths = os.listdir("full_images")
    progress = 0
    for path in image_paths:
        full_path = "full_images/"+path
        cone_estimation(full_path, demo=False)
        progress+=1
        print(f"{progress}/{len(image_paths)}")

if __name__ == '__main__':
    calc_cone_estimates_for_dir()