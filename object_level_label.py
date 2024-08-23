import cv2
import numpy as np
import os

def gaussian_blur(image, kernel_size, sigma):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred

def l0_smoothing(image, lambda_value=0.02):
    img_float32 = np.float32(image) / 255.0
    out = cv2.ximgproc.guidedFilter(img_float32, img_float32, 60, lambda_value)
    out = np.uint8(out * 255.0)
    return out

def l1_smoothing(image, lambda_value=0.1, num_iter=100):
    smoothed = image.copy()
    for _ in range(num_iter):
        gradient_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
        smoothed = cv2.absdiff(image, lambda_value * (gradient_x + gradient_y))
    return smoothed

def L0Smoothing(I):
    def psf2otf_Dy(outSize = None):
        psf = np.zeros((outSize),dtype='float32')
        psf[0, 0] = - 1
        psf[-1, 0] = 1
        otf = np.fft.fft2(psf)
        return otf
    def psf2otf_Dx(outSize = None):
        psf = np.zeros((outSize),dtype='float32')
        psf[0, 0] = - 1
        psf[0, -1] = 1
        otf = np.fft.fft2(psf)
        return otf
    def doL0Smoothing(Im=None, lambda_=None, kappa=None):
        # L0Smooth - Image Smoothing via L0 Gradient Minimization
        #   S = L0Smooth(Im, lambda, kappa) performs L0 graidient smoothing of input
        #   image Im, with smoothness weight lambda and rate kappa.

        #   Paras:
        #   @Im    : Input UINT8 image, both grayscale and color images are acceptable.
        #   @lambda: Smoothing parameter controlling the degree of smooth. (See [1])
        #            Typically it is within the range [1e-3, 1e-1], 2e-2 by default.
        #   @kappa : Parameter that controls the rate. (See [1])
        #            Small kappa results in more iteratioins and with sharper edges.
        #            We select kappa in (1, 2].
        #            kappa = 2 is suggested for natural images.

        #   Example
        #   ==========
        #   Im  = imread('pflower.jpg');
        #   S  = L0Smooth(Im); # Default Parameters (lambda = 2e-2, kappa = 2)
        #   figure, imshow(Im), figure, imshow(S);
        S=Im
        if  (kappa == None):
            kappa = 2.0

        if (lambda_ == None):
            lambda_ = 0.02

        betamax = 100000.0
        fx = np.array([1, - 1])
        fy = np.array([[1], [- 1]])
        N, M, D = Im.shape
        sizeI3D = np.array([N, M, D])
        sizeI2D = np.array([N, M])
        otfFx = psf2otf_Dx(sizeI2D)
        otfFy = psf2otf_Dy(sizeI2D)
        Normin1=np.zeros((N,M,D),dtype=complex)
        for _dim in range(D):
            Normin1[:,:,_dim]=np.fft.fft2(Im[:,:,_dim])
        Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
        if D > 1:
            Denormin2 = np.tile(Denormin2, (D, 1, 1))
            Denormin2 = Denormin2.transpose((1, 2, 0))

        beta = 2 * lambda_
        while beta < betamax:

            Denormin = 1 + beta * Denormin2
            # h-v subproblem
            u01 = Im[:, 0, :] - Im[:, -1, :]
            u01 = np.reshape(u01, (u01.shape[0], 1, u01.shape[1]))
            h = np.concatenate((np.diff(Im, 1, 1), u01), axis=1)
            u10 = Im[0, :, :] - Im[-1, :, :]
            u10 = np.reshape(u10, (1, u10.shape[0], u10.shape[1]))
            v = np.concatenate((np.diff(Im, 1, 0), u10), axis=0)
            if D == 1:
                t = (h ** 2 + v ** 2) < lambda_ / beta
            else:
                t = np.sum((h ** 2 + v ** 2), 3 - 1) < lambda_ / beta
                t = np.tile(t, (D, 1, 1))
                t = t.transpose((1, 2, 0))
                #t = np.matlib.repmat(t, np.array([1, 1, D]))
            h[t] = 0
            v[t] = 0
            # S subproblem
            mu_h01 = h[:, -1, :] - h[:, 0, :]
            mu_h01 = np.reshape(mu_h01, (mu_h01.shape[0], 1, mu_h01.shape[1]))
            Normin2_h = np.concatenate((mu_h01, - np.diff(h, 1, 1)), axis=1)
            mu_v01 = v[-1, :, :] - v[0, :, :]
            mu_v01 = np.reshape(mu_v01, (1, mu_v01.shape[0], mu_v01.shape[1]))
            Normin2_v = np.concatenate((mu_v01, - np.diff(v, 1, 0)), axis=0)
            Normin2 = Normin2_h+Normin2_v
            FS=np.zeros((N,M,D),dtype=complex)
            for _dim in range (D):
                FS[:,:,_dim] = (Normin1[:,:,_dim] + beta * (np.fft.fft2(Normin2[:,:,_dim]))) / Denormin[:,:,_dim]
            #S=np.zeros((N,M,D))
            for _dim in range (D):
                S[:,:,_dim] = np.real(np.fft.ifft2(FS[:,:,_dim]))
    #        FS = (Normin1 + beta * fft2(Normin2)) / Denormin
    #        S = real(ifft2(FS))
            beta = beta * kappa
        return S
    I=np.array(I,dtype='float64')
    I=I/255
    S = doL0Smoothing(I,0.01)
    S=S*255
    return S.astype(np.uint8)

def save_edges(img_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    original_folder = img_path
    small_folder = save_path

    image_files = [f for f in os.listdir(original_folder) if f.endswith(".png")]
    image_files = sorted(image_files)
    # 定义小图的宽度和高度
    # 将彩色图像转换为灰度图像

    for image_file in image_files:

        img = cv2.imread(original_folder+image_file)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        kernel_edge = np.ones((3, 3), np.uint8)

        # 执行膨胀操作
        gray_img = cv2.dilate(gray_img, kernel_edge, iterations=1)

        # 获取非零像素的坐标
        nonzero_points = np.transpose(np.nonzero(gray_img))

        if(len(nonzero_points) <= 2):
            print(image_file)
        # 将坐标列表转换为NumPy数组
        points = nonzero_points.tolist()

        # print(points)
        np.save(os.path.join(save_path+"obj"+image_file[:-4]+".jpg.npy"), points)

def detect_edges(image_path, threshold1=100, threshold2=200):
    # 加载图像
    image = image_path
    # 使用Canny算法进行边缘检测
    edges = cv2.Canny(image, threshold1, threshold2)

    return edges

folder_path = './data/coco/images_v2/val2017/'
img_save_path = './data/coco/images_obj/val2017/'
canny_save_path = './data/coco/images_obj/val_edge_2017/'
labels_save_path = './data/coco/labels_v2/val2017/'

if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)
if not os.path.exists(canny_save_path):
    os.makedirs(canny_save_path)
# 遍历文件夹中的所有图片

count = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 读取灰度图像
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path,1)

        # 增大模糊程度
        kernel_size = 21  # 调整高斯核的大小
        sigma = 2.0  # 调整高斯核的标准差

        # 高斯模糊处理
        blurred_image = gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)

        # 进行L0平滑处理
        smoothed_image = L0Smoothing(blurred_image)
        cann_res = detect_edges(smoothed_image)
        # 保存处理后的图像
        #output_path = os.path.join(folder_path, 'processed', filename)

        cv2.imwrite(os.path.join(img_save_path, filename[:-4]+".png"), smoothed_image)
        cv2.imwrite(os.path.join(canny_save_path, filename[:-4]+".png"), cann_res)
        count = count + 1 
        print(f"Processed: {count}")
        
save_edges(canny_save_path,labels_save_path)
print("All images processed.")