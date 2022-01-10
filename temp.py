import os
import moviepy.editor as moviepy
import shutil

def convert_to_mp4(source_path, destination_path):
    if not os.path.exists(source_path):
        print("Source path doesn't exists.")
        return
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

def convert_to_mp4(source_path, destination_path, delete_folders=False):
    if not os.path.exists(source_path):
        print("Source path doesn't exists.")
        return
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    files = os.listdir(source_path)
    for file in files:
        if os.path.isdir(os.path.join(source_path, file)):
            convert_to_mp4(os.path.join(source_path, file), os.path.join(destination_path, file))
        elif file.find(".avi") != -1:
            clip = moviepy.VideoFileClip(os.path.join(source_path, file))
            print("writing: ", file)
            file_name = file.split(".avi")[0]
            clip.write_videofile(os.path.join(destination_path, file_name+".mp4"))

def delete_avi(path):
    files = os.listdir(path)
    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            delete_avi(os.path.join(path, file))
        elif file.find(".avi") != -1:
            print("Deleting: ", os.path.join(path, file))
            os.remove(os.path.join(path, file))

def delete_unwanted_folders(path):
    files = os.listdir(path)
    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            copy_and_delete_folder(os.path.join(path, file))

def copy_and_delete_folder(path):
    folders = os.listdir(path)
    for folder in folders:
        if os.path.isdir(os.path.join(path, folder)):
            files = os.listdir(os.path.join(path, folder))
            for file in files:
                if not os.path.isdir(os.path.join(path, folder, file)):
                    shutil.move(os.path.join(path, folder, file), os.path.join(path,  file))
            os.rmdir(os.path.join(path, folder))
            print("removed: ", os.path.join(path, folder))





source_path = "/home/kiran/kiran/Thesis/code/VideoMoCo/dataset/UCF11"
destination_path = "/home/kiran/kiran/Thesis/code/VideoMoCo/dataset/UCF11_new"
# convert_to_mp4(source_path, destination_path)
#delete_avi(source_path, destination_path)
delete_unwanted_folders("/home/kiran/kiran/Thesis/code/Video_fusion/datasets/UCF11/train")


'''
a = np.array(images[0][0, :, 0, :, :])
b = np.array(images[1][0, :, 0, :, :])
c = np.array(images[2][0, :, 0, :, :])
a = np.moveaxis(a, 0, -1)
b = np.moveaxis(b, 0, -1)
c = np.moveaxis(c, 0, -1)
plt.imshow(a)
plt.show()
plt.imshow(b)
plt.show()
plt.imshow(c)
plt.show()


--dist-url tcp://localhost:10055 --multiprocessing-distributed --world-size 1 --rank 0 --fix-pred-lr --resume /home/kiran/kiran/Thesis/code/kiran_code/checkpoint_0006.pth.tar --start-epoch 7
/home/kiran/kiran/Thesis/code/Video_fusion/datasets/UCF11/train/basketball/v_shooting_24_01.mpg 1 8
'''