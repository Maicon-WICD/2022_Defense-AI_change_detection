import os

noise_img = ['2017_KAG_2LB_000912.png', '2019_KBG_3LB_000073.png', '2019_KDG_2LB_000871.png', '2019_WSN_2LB_000047.png', '2019_JNG_1LB_000005.png', '2019_KSG_2LB_000374.png', '2019_SMG_1LB_000027.png']

noise_2n3 = ['2015_DMG_2LB_000436.png', '2016_YDP_JJG_000151.png', '2017_KSG_SAG_000148.png', '2017_YDP_2LB_000503.png', '2018_KAG_SAG_000041.png', '2018_KSG_JJG_000007.png', '2018_SMG_3LB_000185.png', '2018_SPG_3LB_000326.png', '2019_JNG_SAG_000234.png', '2019_JRG_SAG_000081.png', '2019_KSG_2LB_000544.png', '2019_KSG_SAG_000123.png', '2019_MPG_3LB_000229.png']

noise_1n2 = ['2016_YDP_JJG_000075.png', '2018_SCG_2LB_000288.png', '2018_SCG_2LB_000292.png', '2019_JNG_1LB_000129.png', '2019_KSG_2LB_000589.png', '2019_KSG_KNI_000461.png']
data_list = [noise_img,noise_2n3,noise_1n2]
file_list = ['x','y']
for noise in range(len(data_list)):
    for i in data_list[noise]:
        for k in file_list:
            file_path='/data/train/'+k+'/'+i
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(file_path)