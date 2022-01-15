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



TRAIN video number: 1599
TRAIN video number: 1119, VAL video number: 319, Test video number: 161.
Epoch: 0 -> Epoch time: 349.90 s. Training loss: 2.310171440882342 , Training acc: 0.17605004468275245 => Validation loss: 4.315281790526586, Validation acc: 0.21003134796238246
Epoch: 1 -> Epoch time: 357.96 s. Training loss: 2.0699696467391084 , Training acc: 0.2806076854334227 => Validation loss: 5.154357326441096, Validation acc: 0.30094043887147337
Epoch: 2 -> Epoch time: 359.56 s. Training loss: 1.9361990291093076 , Training acc: 0.3351206434316354 => Validation loss: 3.4768249334709234, Validation acc: 0.3166144200626959
Epoch: 3 -> Epoch time: 361.57 s. Training loss: 1.7837179770427092 , Training acc: 0.3932082216264522 => Validation loss: 3.1835388594772667, Validation acc: 0.3605015673981191
Epoch: 4 -> Epoch time: 362.80 s. Training loss: 1.7171373241980161 , Training acc: 0.4307417336907953 => Validation loss: 3.9598308356748704, Validation acc: 0.3887147335423197
Epoch: 5 -> Epoch time: 364.18 s. Training loss: 1.5944953602871725 , Training acc: 0.4673815907059875 => Validation loss: 3.9264287188183515, Validation acc: 0.3824451410658307
Epoch: 6 -> Epoch time: 364.42 s. Training loss: 1.4823371923129474 , Training acc: 0.49776586237712245 => Validation loss: 2.839299757672234, Validation acc: 0.4043887147335423
Epoch: 7 -> Epoch time: 366.04 s. Training loss: 1.3878542266253915 , Training acc: 0.550491510277033 => Validation loss: 3.0994158345747564, Validation acc: 0.47335423197492166
Epoch: 8 -> Epoch time: 371.63 s. Training loss: 1.2787630725718502 , Training acc: 0.5755138516532619 => Validation loss: 2.8037274994072505, Validation acc: 0.4702194357366771
Epoch: 9 -> Epoch time: 368.94 s. Training loss: 1.1704594045478318 , Training acc: 0.613941018766756 => Validation loss: 4.310425952387414, Validation acc: 0.45768025078369906
Epoch: 10 -> Epoch time: 366.91 s. Training loss: 1.1300515267898195 , Training acc: 0.6282394995531725 => Validation loss: 2.2296707805817277, Validation acc: 0.5266457680250783
Epoch: 11 -> Epoch time: 370.33 s. Training loss: 1.04960303547393 , Training acc: 0.6657730116175157 => Validation loss: 2.7169738398726624, Validation acc: 0.5642633228840125
Epoch: 12 -> Epoch time: 368.31 s. Training loss: 1.0041643311435888 , Training acc: 0.6916890080428955 => Validation loss: 3.170990001762766, Validation acc: 0.5266457680250783
Epoch: 13 -> Epoch time: 374.20 s. Training loss: 0.9247782122210733 , Training acc: 0.7122430741733691 => Validation loss: 2.3032196536732954, Validation acc: 0.5987460815047022
Epoch: 14 -> Epoch time: 371.88 s. Training loss: 0.877403765494403 , Training acc: 0.7363717605004468 => Validation loss: 2.2734093796305386, Validation acc: 0.6175548589341693
Epoch: 15 -> Epoch time: 368.76 s. Training loss: 0.8025891926854716 , Training acc: 0.7542448614834674 => Validation loss: 1.9459389884097618, Validation acc: 0.6081504702194357
Epoch: 16 -> Epoch time: 366.35 s. Training loss: 0.7936091150689338 , Training acc: 0.7444146559428061 => Validation loss: 2.4374427115311845, Validation acc: 0.6332288401253918
Epoch: 17 -> Epoch time: 363.14 s. Training loss: 0.6781930075376295 , Training acc: 0.7935656836461126 => Validation loss: 2.1273478022976633, Validation acc: 0.6363636363636364
Epoch: 18 -> Epoch time: 367.69 s. Training loss: 0.6677699299312995 , Training acc: 0.8025022341376229 => Validation loss: 2.4947598917246068, Validation acc: 0.6332288401253918
Epoch: 19 -> Epoch time: 369.11 s. Training loss: 0.6301424175218147 , Training acc: 0.806970509383378 => Validation loss: 3.560758187695501, Validation acc: 0.6332288401253918
Epoch: 20 -> Epoch time: 368.14 s. Training loss: 0.6419344616538313 , Training acc: 0.80875781948168 => Validation loss: 4.282577112404397, Validation acc: 0.5830721003134797
Epoch: 21 -> Epoch time: 363.71 s. Training loss: 0.5633418798163933 , Training acc: 0.8355674709562109 => Validation loss: 3.275263464057571, Validation acc: 0.6144200626959248
Epoch: 22 -> Epoch time: 366.11 s. Training loss: 0.5279095007256339 , Training acc: 0.839142091152815 => Validation loss: 1.9220847609049088, Validation acc: 0.6426332288401254
Epoch: 23 -> Epoch time: 368.15 s. Training loss: 0.5156669010275177 , Training acc: 0.8436103663985701 => Validation loss: 2.5513596717449216, Validation acc: 0.6520376175548589
Epoch: 24 -> Epoch time: 367.56 s. Training loss: 0.4508417592192667 , Training acc: 0.8614834673815907 => Validation loss: 1.6622647606354803, Validation acc: 0.6802507836990596
Epoch: 25 -> Epoch time: 365.87 s. Training loss: 0.44093624484043437 , Training acc: 0.870420017873101 => Validation loss: 5.092146703987964, Validation acc: 0.6144200626959248
Epoch: 26 -> Epoch time: 366.54 s. Training loss: 0.44674421850941143 , Training acc: 0.8668453976764968 => Validation loss: 1.7487688378855182, Validation acc: 0.6896551724137931
 Epoch: 27 -> Epoch time: 369.34 s. Training loss: 0.37918150497633696 , Training acc: 0.8865058087578195 => Validation loss: 2.4993028817269076, Validation acc: 0.6802507836990596
Epoch: 28 -> Epoch time: 367.80 s. Training loss: 0.40551875206708377 , Training acc: 0.8748882931188561 => Validation loss: 1.6883565462736443, Validation acc: 0.6739811912225705
Epoch: 29 -> Epoch time: 367.08 s. Training loss: 0.37999182127449393 , Training acc: 0.8856121537086684 => Validation loss: 2.439140536551349, Validation acc: 0.6739811912225705
Epoch: 30 -> Epoch time: 366.51 s. Training loss: 0.35053617648248164 , Training acc: 0.8972296693476318 => Validation loss: 1.7256184816396853, Validation acc: 0.677115987460815
Epoch: 31 -> Epoch time: 369.12 s. Training loss: 0.31219920408620966 , Training acc: 0.902591599642538 => Validation loss: 1.773602836025384, Validation acc: 0.677115987460815
Epoch: 32 -> Epoch time: 364.66 s. Training loss: 0.3008726037681169 , Training acc: 0.9133154602323503 => Validation loss: 2.727018230105614, Validation acc: 0.658307210031348
Epoch: 33 -> Epoch time: 367.32 s. Training loss: 0.28375019132093127 , Training acc: 0.9285075960679178 => Validation loss: 2.512496356981137, Validation acc: 0.6896551724137931
Epoch: 34 -> Epoch time: 368.06 s. Training loss: 0.2860401760227562 , Training acc: 0.9115281501340483 => Validation loss: 1.8129053208866366, Validation acc: 0.7021943573667712
Epoch: 35 -> Epoch time: 367.46 s. Training loss: 0.2752774540426409 , Training acc: 0.9231456657730116 => Validation loss: 2.726426328504749, Validation acc: 0.6739811912225705
Epoch: 36 -> Epoch time: 369.16 s. Training loss: 0.2778097562263221 , Training acc: 0.9213583556747096 => Validation loss: 1.2916673662909033, Validation acc: 0.6896551724137931
Epoch: 37 -> Epoch time: 368.63 s. Training loss: 0.22517909799789776 , Training acc: 0.9499553172475425 => Validation loss: 2.041962075580159, Validation acc: 0.7021943573667712
Epoch: 38 -> Epoch time: 370.74 s. Training loss: 0.2468128069515972 , Training acc: 0.9329758713136729 => Validation loss: 1.6426077253381208, Validation acc: 0.7241379310344828



Sim Siam pretrained
Epoch: 46 -> Epoch time: 4.25 s. Training loss: 1.697538354434073 , Training acc: 0.4575513851653262 => Validation loss: 1.9268481450155377, Validation acc: 0.3448275862068966
Epoch: 47 -> Epoch time: 4.25 s. Training loss: 1.6882079676858017 , Training acc: 0.450402144772118 => Validation loss: 1.900357673689723, Validation acc: 0.3667711598746082
Epoch: 48 -> Epoch time: 4.42 s. Training loss: 1.6846598401366333 , Training acc: 0.45933869526362825 => Validation loss: 1.891060078330338, Validation acc: 0.36363636363636365
Epoch: 49 -> Epoch time: 4.32 s. Training loss: 1.6841641875516091 , Training acc: 0.46380697050938335 => Validation loss: 1.9396673817187549, Validation acc: 0.36363636363636365
Epoch: 50 -> Epoch time: 4.31 s. Training loss: 1.6798256536413516 , Training acc: 0.4539767649687221 => Validation loss: 1.8926901875063777, Validation acc: 0.3605015673981191
Epoch: 51 -> Epoch time: 4.40 s. Training loss: 1.6747823015986276 , Training acc: 0.4459338695263628 => Validation loss: 1.9037524458020925, Validation acc: 0.3605015673981191


Epoch: 0 -> Epoch time: 333.66 s. Training loss: 2.333566065984113 , Training acc: 0.14655942806076855 => Validation loss: 2.2910089291632176, Validation acc: 0.19435736677115986
100%|██████████| 560/560 [04:17<00:00,  2.18it/s]
Epoch: 1 -> Epoch time: 332.63 s. Training loss: 2.3626331606081554 , Training acc: 0.14209115281501342 => Validation loss: 2.4317082807421686, Validation acc: 0.14420062695924765
100%|██████████| 560/560 [04:29<00:00,  2.08it/s]
  0%|          | 0/560 [00:00<?, ?it/s]Epoch: 2 -> Epoch time: 355.01 s. Training loss: 2.318031190974372 , Training acc: 0.17515638963360142 => Validation loss: 2.1525632705539466, Validation acc: 0.20376175548589343

TRAIN video number: 1119, VAL video number: 319, Test video number: 161.
 14%|█▎        | 76/560 [00:35<03:23,  2.37it/s]/home/kiran/kiran/Thesis/code/Video_fusion/datasets/UCF11/train/basketball/v_shooting_16_05.mpg
 74%|███████▍  | 415/560 [03:05<01:08,  2.13it/s]/home/kiran/kiran/Thesis/code/Video_fusion/datasets/UCF11/train/horse_riding/v_riding_15_07.mpg
 96%|█████████▋| 540/560 [04:01<00:08,  2.33it/s]/home/kiran/kiran/Thesis/code/Video_fusion/datasets/UCF11/train/basketball/v_shooting_25_06.mpg
 
 
 0%|          | 0/559 [00:00<?, ?it/s][W reducer.cpp:1303] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
100%|██████████| 559/559 [04:30<00:00,  2.07it/s]
Epoch: 0 -> Epoch time: 344.86 s. Training loss: 2.3411578854849173 , Training acc: 0.1593554162936437 => Validation loss: 4.303802877664566, Validation acc: 0.21316614420062696
100%|██████████| 559/559 [04:32<00:00,  2.05it/s]
Epoch: 1 -> Epoch time: 347.78 s. Training loss: 2.1304291932442108 , Training acc: 0.24529991047448524 => Validation loss: 2.265936866030097, Validation acc: 0.2946708463949843
100%|██████████| 559/559 [04:32<00:00,  2.05it/s]
  0%|          | 0/559 [00:00<?, ?it/s]Epoch: 2 -> Epoch time: 347.99 s. Training loss: 1.9898952616775185 , Training acc: 0.3169203222918532 => Validation loss: 2.327288646157831, Validation acc: 0.322884012539185
100%|██████████| 559/559 [04:34<00:00,  2.03it/s]
Epoch: 3 -> Epoch time: 350.41 s. Training loss: 1.8463866481414208 , Training acc: 0.34825425246195163 => Validation loss: 2.5835814754478634, Validation acc: 0.3573667711598746
100%|██████████| 559/559 [04:34<00:00,  2.04it/s]
Epoch: 4 -> Epoch time: 350.62 s. Training loss: 1.6824950203166238 , Training acc: 0.4252461951656222 => Validation loss: 2.535973772930447, Validation acc: 0.4012539184952978
100%|██████████| 559/559 [04:35<00:00,  2.03it/s]

single frame diff:
0%|          | 0/559 [00:00<?, ?it/s][W reducer.cpp:1303] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
100%|██████████| 559/559 [04:35<00:00,  2.03it/s]
Epoch: 0 -> Epoch time: 349.80 s. Training loss: 2.338510570138 , Training acc: 0.1459265890778872 => Validation loss: 2.3287020295858385, Validation acc: 0.15673981191222572
100%|██████████| 559/559 [04:38<00:00,  2.01it/s]
  0%|          | 0/559 [00:00<?, ?it/s]Epoch: 1 -> Epoch time: 353.24 s. Training loss: 2.246197736967015 , Training acc: 0.21217547000895254 => Validation loss: 2.5252424627542496, Validation acc: 0.16927899686520376
100%|██████████| 559/559 [04:36<00:00,  2.02it/s]
Epoch: 2 -> Epoch time: 350.82 s. Training loss: 2.1762871833947988 , Training acc: 0.23187108325872874 => Validation loss: 2.414639899134636, Validation acc: 0.15673981191222572
100%|██████████| 559/559 [04:49<00:00,  1.93it/s]
Epoch: 3 -> Epoch time: 371.51 s. Training loss: 2.1147545282870586 , Training acc: 0.24619516562220234 => Validation loss: 2.1053408019244673, Validation acc: 0.21630094043887146

'''