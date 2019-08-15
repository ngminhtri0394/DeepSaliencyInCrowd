imgs_train_path = 'data/img/'
maps_train_path = 'data/map/'
fixs_train_path = 'data/fixation/'

imgs_test_path = 'data/img/'
maps_test_path = 'data/map/'
fixs_test_path = 'data/fixation/'

imgs_dev_path = 'data/img_val/'
maps_dev_path = 'data/map_val/'
fixs_dev_path = 'data/fixation_val/'

net = 'msdensenet_att'
run_salicon_dataset = 0
# batch size
b_s = 1
# number of rows of input images
shape_r = 960
# number of cols of input images
shape_c = 1280
# number of rows of downsampled maps
shape_r_gt = 30
# number of cols of downsampled maps
shape_c_gt = 40
# number of rows of model outputs
shape_r_out = 480
# number of cols of model outputs
shape_c_out = 640
# final upsampling factor
upsampling_factor = 16
# number of epochs
nb_epoch = 150
# number of timestep
nb_timestep = 4
# number of learned priors
nb_gaussian = 16
# number of training images
nb_imgs_train = 450
# number of validation images
nb_imgs_val = 80
#image final output width
shape_r_fout = 768
#image final output height
shape_c_fout = 1024
#post process gaussian sigma
gaussian_sigma = 6
