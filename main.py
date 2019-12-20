from utils import load_img, memory, data_augmentation, load_folder_img

import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K_
from tensorflow import keras

import os
import cv2
import time, random
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


import glob
import gc; gc.enable()
from skimage.segmentation import mark_boundaries
from skimage.util import montage
from skimage.io import imread
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)


class Pix2Pix():
    def __init__(self,
                start_epoch,
                lambda_value,
                learning_rate,
                n_discriminator,
                input_channels,
                output_channels,
                generator_activation,
                testsavePath):

        self.epoch = start_epoch
        self.shape = input_channels[0]
        self.lambda_value = lambda_value
        self.n_discriminator = n_discriminator
        self.testsavePath = testsavePath
        if generator_activation == 'elu':
            self.generator_activation = tf.keras.activations.elu
        elif generator_activation == 'leaky_relu':
            self.generator_activation = tf.nn.leaky_relu
        elif generator_activation == 'relu':
            self.generator_activation = tf.keras.activations.relu
        else:
            self.generator_activation = generator_activation
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.build_patch_discriminator(input_channels, output_channels, False)
        self.build_unet_generator(input_channels, output_channels)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = 0.5, epsilon=1e-6)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = 0.5, epsilon=1e-6)

        self.g_total_losses = []
        self.gen_losses     = []
        self.l1_losses      = []

        self.d_total_losses = []
        self.d_real_losses = []
        self.d_fake_losses = []


        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = 'logs/gradient_tape/' + self.current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)


    def downsample(self, filters, size, apply_norm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                            kernel_initializer=initializer,
                            # kernel_regularizer=keras.regularizers.l2(0.01),
                            use_bias=False))

        if apply_norm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        # kernel_regularizer=keras.regularizers.l2(0.01),
                                        use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def build_patch_discriminator(self, input_channel, target_channel, _input=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        tar = tf.keras.layers.Input(shape=target_channel, name='target_image')
        inp = tf.keras.layers.Input(shape=input_channel, name='input_image')
        x = tf.keras.layers.concatenate([tar, inp])  # (bs, 512, 512, channels*2)

        down1 = self.downsample(64, 4, False)(x)  # (bs, 256, 256, 64)
        down2 = self.downsample(128, 4)(down1)  # (bs, 128, 128, 128)
        down3 = self.downsample(256, 4)(down2)  # (bs, 64, 64, 256)
        # down4 = self.downsample(512, 4)(down3)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(
            512, 4, strides=1, kernel_initializer=initializer,
            use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        norm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(
            1, 4, strides=1,
            kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        self.discriminator = tf.keras.Model(inputs=[tar, inp], outputs=last, name="Discriminator")
        

    def build_unet_generator(self, input_channels, output_channels):
        down_stack = [
        self.downsample(64, 4, apply_norm=False),   # (bs, 256, 256, 64)
        self.downsample(128, 4, apply_norm=False),  # (bs, 128, 128, 128)
        self.downsample(256, 4),                    # (bs, 64, 64, 256)
        self.downsample(512, 4),                    # (bs, 32, 32, 256)
        self.downsample(512, 4),                    # (bs, 16, 16, 256)
        self.downsample(512, 4),                    # (bs, 8, 8, 512)
        self.downsample(512, 4),                    # (bs, 4, 4, 512)
        self.downsample(512, 4)                     # (bs, 2, 2, 512)
        ]

        up_stack = [
        self.upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 512)
        self.upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 512)
        self.upsample(512, 4, apply_dropout=True),  # (bs, 16, 16, 512)
        self.upsample(512, 4, apply_dropout=True),  # (bs, 32, 32, 512)
        self.upsample(256, 4),  # (bs, 64, 64, 256)
        self.upsample(128, 4),  # (bs, 128, 128, 256)
        self.upsample(64, 4)  # (bs, 256, 256, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(
            output_channels[2], 4, strides=2,
            padding='same', kernel_initializer=initializer,
            activation=self.generator_activation)  # (bs, 512, 512, 3)

        concat = tf.keras.layers.Concatenate()
        inputs = tf.keras.layers.Input(shape=input_channels, name='new_frame')
        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])
        x = last(x)
        
        self.unet_generator =  tf.keras.Model(inputs=inputs, outputs=x, name="Generator")

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(
            tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(
            tf.zeros_like(disc_generated_output), disc_generated_output)

        total_loss = real_loss + generated_loss
        return total_loss, real_loss, generated_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        gen_loss = self.loss_object(
            tf.ones_like(disc_generated_output), disc_generated_output)

        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_loss  = gen_loss + self.lambda_value * l1_loss
        return total_loss, gen_loss, l1_loss

    def testV(self, epoch, filepath):
        def process_image(image):
            # NOTE: The output you return should be a color image (3 channel) for processing video below
            # TODO: put your pipeline here,
            # you should return the final output (image where lines are drawn on lanes)
            image_shape = image.shape[:2]
        #     print(image_shape)
            image = cv2.resize(image,(256,256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            first_img = np.expand_dims(image, 0)/255.0
        #     result = image_pipeline(image)
            first_seg = self.unet_generator.predict(first_img)
            first_img[0][:,:,0] = first_img[0][:,:,0]*0.7 + 0.3*first_seg[0, :, :, 0]
            result = np.array(np.clip(first_img[0]*255,0,255),dtype=np.float)
        #     print(image_shape[:2],result.shape,type(result[0][0][0]))
            result = cv2.resize(result,image_shape[::-1])
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        #     result = result[...,::-1]
            return result

        filenames = os.listdir(filepath)
        for filename in filenames:
            path = os.path.join(filepath, filename)
            clip = VideoFileClip(path)
            white_clip = clip.fl_image(process_image)

            savePath = os.path.join(filepath, 'prediction_Videos')
            savePath = os.path.join(savePath, filename)
            white_clip.write_videofile(savePath + '{}_detection.mp4'.format(epoch), audio=False)

    def videoTest(self, epoch, videopath, size, _gray):
        video_list = os.listdir(videopath)
        for video in video_list:
            savePath = os.path.join(self.testsavePath, str(epoch).zfill(4))
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            savePath = os.path.join(savePath, video)
            if not os.path.exists(savePath):
                os.mkdir(savePath)

            cnt = 1
            video_path = os.path.join(videopath, video)
            print("Video path: {}".format(video_path))

            cap = cv2.VideoCapture(video_path)
            frame_cnt = 0
            while True:
                if frame_cnt == 200:
                    break

                _, frame = cap.read()
                if not _:
                    print("Video ends")
                    break

                if _gray:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2BGR)
                else:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, dsize=(size, size))
                rgb_frame = (rgb_frame.astype(np.float32)/255.)
                resized = tf.expand_dims(rgb_frame, 0)

                gen_img = self.unet_generator.predict(resized)
                gen_img = tf.squeeze(gen_img, 0)

                plt.figure(figsize=(15, 15))
                plt.subplot(1, 2, 1)
                plt.title("Original")
                plt.imshow(tf.keras.preprocessing.image.array_to_img(rgb_frame))
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.title("Predicted")
                plt.imshow(tf.keras.preprocessing.image.array_to_img(gen_img), cmap='gray')
                plt.axis('off')

                plt.savefig(os.path.join(savePath, str(cnt).zfill(6)) + '.png', dpi=200)
                plt.close()
                cnt += 1
                frame_cnt += 1

            if cap.isOpened():
                cap.release()

    def testImgs(self, result_num, epoch, test_steps, test_flow):
        print("Test During the Training")
        test_step = 1
        for test_img in tqdm(test_flow, total=test_steps):
            if test_step > test_steps:
                break

            result   = self.unet_generator.predict(test_img)
            # test_img = np.array(test_img, dtype=np.float32)
            # result   = np.array(result, dtype=np.float32) 
            
            batch_rgb = montage_rgb(test_img)
            batch_seg = montage(result[:, :, :, 0])
            plot_img  = mark_boundaries(batch_rgb, batch_seg.astype(int))

            test_img = tf.squeeze(test_img, 0)
            result   = tf.squeeze(result, 0)

            plt.figure(figsize=(15, 15))
            plt.subplot(1, 3, 1)
            plt.title("Original")
            plt.imshow(tf.keras.preprocessing.image.array_to_img(test_img))
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Predicted")
            plt.imshow(tf.keras.preprocessing.image.array_to_img(result), cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Predicted")
            plt.imshow(tf.keras.preprocessing.image.array_to_img(plot_img))
            plt.axis('off')
        
            savePath = os.path.join("C:\\Users\H\Desktop\ADD\Result_imgs", str(epoch+1).zfill(4))
            if not os.path.exists(savePath):
                os.mkdir(savePath)

            plt.savefig(os.path.join(savePath, str(result_num).zfill(8)) + '.png')
            result_num += 1
            plt.close()

            test_step+=1
        print("Done")

    # def train_step(self, epoch, imag, target):
    #     with tf.GradientTape(watch_accessed_variables=False) as gen_tape:
            
    #         """ 
    #             Train discriminator [n_discriminator] times
    #         """
    #         self.unet_generator.trainable = False
    #         self.discriminator.trainable = True
    #         for _ in range(self.n_discriminator):
    #             with tf.GradientTape(watch_accessed_variables=False) as dis_tape:
    #                 dis_tape.watch(self.discriminator.trainable_variables)

    #                 fake_img = self.unet_generator(imag, training=True)
    #                 disc_real_output = self.discriminator([target, imag], training=True)
    #                 disc_generated_output = self.discriminator([fake_img, imag], training=True)
    #                 disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
    #             discriminator_gradients = dis_tape.gradient(disc_loss[0], self.discriminator.trainable_variables)
    #             self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

    #             with self.train_summary_writer.as_default():
    #                 tf.summary.scalar('D Total Loss', disc_loss[0].numpy(), step=epoch)
    #                 tf.summary.scalar('D Real Loss', disc_loss[1].numpy(), step=epoch)
    #                 tf.summary.scalar('D Fake Loss', disc_loss[2].numpy(), step=epoch)
    #             del dis_tape

    #         self.unet_generator.trainable = True
    #         self.discriminator.trainable = False
    #         gen_tape.watch(self.unet_generator.trainable_variables)
    #         fake_img = self.unet_generator(imag, training=True)
    #         disc_real_output = self.discriminator([target, imag], training=True)
    #         disc_generated_output = self.discriminator([fake_img, imag], training=True)
    #         gen_loss = self.generator_loss(disc_generated_output, fake_img, target)
    #     generator_gradients = gen_tape.gradient(gen_loss[0], self.unet_generator.trainable_variables)
    #     self.generator_optimizer.apply_gradients(zip(generator_gradients, self.unet_generator.trainable_variables))


    #     with self.train_summary_writer.as_default():
    #         tf.summary.scalar('G Total Loss', gen_loss[0].numpy(), step=epoch)
    #         tf.summary.scalar('G Fake Loss', gen_loss[1].numpy(), step=epoch)
    #         tf.summary.scalar('G L1 Loss', gen_loss[2].numpy(), step=epoch)
    #         # tf.summary.histogram("Generator params", self.unet_generator.trainable_variables)
    #         # tf.summary.histogram("Discriminator params", self.discriminator.trainable_variables)


    #     return gen_loss, disc_loss

    def train_step(self, _step, imag, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            fake_img = self.unet_generator(imag, training=True)
            disc_real_output = self.discriminator([imag, target], training=True)
            disc_generated_output = self.discriminator([imag, fake_img], training=True)
            
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
            gen_loss = self.generator_loss(disc_generated_output, fake_img, target)
        discriminator_gradients = dis_tape.gradient(disc_loss[0], self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
        generator_gradients = gen_tape.gradient(gen_loss[0], self.unet_generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.unet_generator.trainable_variables))

        with self.train_summary_writer.as_default():
            tf.summary.scalar('D Total Loss', disc_loss[0].numpy(), step=_step)
            tf.summary.scalar('D Real Loss', disc_loss[1].numpy(), step=_step)
            tf.summary.scalar('D Fake Loss', disc_loss[2].numpy(), step=_step)

            tf.summary.scalar('G Total Loss', gen_loss[0].numpy(), step=_step)
            tf.summary.scalar('G Fake Loss', gen_loss[1].numpy(), step=_step)
            tf.summary.scalar('G L1 Loss', gen_loss[2].numpy(), step=_step)
        return gen_loss, disc_loss
   
    def train(self, epochs, train_steps_for_epoch, test_steps, train_flow, test_Flow, output_Gray=True):
        result_num = 1
        steps      = 0
        for epoch in range(self.epoch, epochs):
            avg_d_total_loss  = 0
            avg_d_real_loss   = 0
            avg_d_fake_loss   = 0
            avg_g_total_loss  = 0
            avg_gen_loss      = 0
            avg_l1_loss       = 0

            # random_seed = random.randint(1, 100)
            # original_flow = train_datagen.flow_from_directory(train_original_img_dir, target_size = size[:2],
            #                                     batch_size=batchS, seed = 1,
            #                                     color_mode='rgb', class_mode=None)
            # mask_flow     = train_datagen.flow_from_directory(train_truth_img_dir, target_size = size[:2],
            #                                             batch_size=batchS, seed = 1,
            #                                             color_mode='grayscale', class_mode=None)
            # train_flow    = zip(original_flow, mask_flow)

            for imag, target in tqdm(train_flow, total=train_steps_for_epoch):
                g_loss, d_loss = self.train_step(steps, imag, target)

                self.d_total_losses.append(d_loss[0])
                self.d_real_losses.append(d_loss[1])
                self.d_fake_losses.append(d_loss[2])
                self.g_total_losses.append(g_loss[0])
                self.gen_losses.append(g_loss[1])
                self.l1_losses.append(g_loss[2])

                avg_d_total_loss += d_loss[0]
                avg_d_real_loss  += d_loss[1]
                avg_d_fake_loss  += d_loss[2]
                avg_g_total_loss += g_loss[0]
                avg_l1_loss      += g_loss[1]
                avg_gen_loss     += g_loss[2]
                steps            += 1

                if steps%train_steps_for_epoch == 0 and steps is not 0:
                    break


            print("Epoch: {}".format(epoch+1))
            print("[Generator Loss] Total: {} (Gen Loss: {}, L1 Loss: {})".format(avg_g_total_loss/steps, avg_gen_loss/steps, avg_l1_loss/steps))
            print("[Discriminator Loss] {} (Real: {}, Fake: {})".format(avg_d_total_loss/steps, avg_d_real_loss/steps, avg_d_fake_loss/steps))

            
            date = time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time()))
            self.discriminator.save("C:\\Users\H\Desktop\ADD\models\Discriminator\{}_{}_discriminator.h5".format(date, str(epoch).zfill(4)))
            self.unet_generator.save("C:\\Users\H\Desktop\ADD\models\Generator\{}_{}_generator.h5".format(date, str(epoch).zfill(4)))

            self.discriminator.save("C:\\Users\H\Desktop\ADD\models\Discriminator\discriminator_{}.h5".format(self.shape))
            self.unet_generator.save("C:\\Users\H\Desktop\ADD\models\Generator\generator_{}.h5".format(self.shape))

            if (epoch+1)%2 == 0:
                self.testImgs(result_num, epoch+1, test_steps, test_flow)

            if (epoch+1)%50 == 0:
                print("Video Test")
                self.testV(epoch + 1, "C:\\Users\H\Desktop\ADD\Test_Videos")

    def load_weights(self):
        self.discriminator.load_weights("C:\\Users\H\Desktop\ADD\models\Discriminator\discriminator_{}.h5".format(self.shape))
        self.unet_generator.load_weights("C:\\Users\H\Desktop\ADD\models\Generator\generator_{}.h5".format(self.shape))


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    config = tf.compat.v1.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    model = Pix2Pix(start_epoch=1, lambda_value=100, learning_rate=2e-4, n_discriminator=0, 
                    input_channels=(256, 256, 3), output_channels=(256, 256, 1), generator_activation='sigmoid',
                    testsavePath = '')

    model.discriminator.summary()
    model.unet_generator.summary()

    train_original_img_dir = ""
    train_truth_img_dir = ""
    test_original_dir = ""
    train_original_img_dir_1 = ""
    train_truth_img_dir_1 = ""
    test_original_dir_1 = ""
    train_original = []; test_original = []; train_truth = []

    size = (256, 256, 3)
    print("Start loading IMGS")
    before_load_imgs = memory()
    img_load_start = time.time()

    train_original = load_img(train_original_img_dir_1, size[:2], False)
    train_truth = load_img(train_truth_img_dir_1, size[:2], False)
    test_original = load_img(test_original_dir_1, size[:2], False)

    img_load_end = time.time()
    print("Finish loading IMGS with", img_load_end-img_load_start, "s")
    print("Imgs memory usage:", memory()-before_load_imgs)

    from keras.preprocessing.image import ImageDataGenerator
    data_args = dict(rescale=1/255.,
                    # featurewise_center=True,
                    # featurewise_std_normalization=True,
                    rotation_range=45,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.02,
                    zoom_range= [0.7, 1.3],
                    brightness_range=[0.6, 1.5],
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode = 'reflect',
                    data_format = 'channels_last'
    )

    train_datagen = ImageDataGenerator(**data_args)
    test_datagen = ImageDataGenerator(rescale=1/255.)


    batchS = 8
    original_flow = train_datagen.flow_from_directory(train_original_img_dir, target_size = size[:2],
                                                batch_size=batchS, seed = 1, shuffle=True,
                                                color_mode='rgb', class_mode=None)
    mask_flow     = train_datagen.flow_from_directory(train_truth_img_dir, target_size = size[:2],
                                                batch_size=batchS, seed = 1, shuffle=True,
                                                color_mode='grayscale', class_mode=None)
    train_flow    = zip(original_flow, mask_flow)

    test_flow     = test_datagen.flow_from_directory(test_original_dir, target_size = size[:2],
                                                    color_mode='rgb',
                                                    batch_size=1, class_mode=None, shuffle=False)

    steps_per_epoch = int(len(train_original) / (batchS))
    test_steps = int(len(test_original))


    # """Data checking"""
    # from keras.preprocessing.image import array_to_img
    # for img, mask in train_flow:
    #     plt.imshow(array_to_img(img[0]))
    #     plt.colorbar()
    #     plt.show()

    #     plt.imshow(array_to_img(mask[0]))
    #     plt.colorbar()
    #     plt.show()

    # model.testV(200, "C:\\Users\H\Desktop\ADD\Test_Videos")
    # print("Video Test")
    # model.videoTest(200, "C:\\Users\H\Desktop\ADD\Test_Videos", 256, False)
    model.train(1000, steps_per_epoch, test_steps, train_flow, test_flow, output_Gray=False)