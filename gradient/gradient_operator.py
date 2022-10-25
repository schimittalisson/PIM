import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class get_color():
    def read_img(self):
        print("**************************************************")
        print("*                     INICIO                     *")
        print("**************************************************")
        img_lua = cv2.imread('./images/Lua1_gray.jpg')
        img_chessboard = cv2.imread('images/chessboard_inv.png')
        img02 = cv2.imread('images/img02.jpg')
        return img_lua, img_chessboard, img02

    def median_filter(data, filter_size):
        temp = []
        indexer = filter_size // 2
        data_final = []
        data_final = np.zeros((len(data),len(data[0])))
        for i in range(len(data)):
            for j in range(len(data[0])):
                for z in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                        for c in range(filter_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(filter_size):
                                temp.append(data[i + z - indexer][j + k - indexer])
                temp.sort()
                data_final[i][j] = temp[len(temp) // 2]
                temp = []
        return data_final

    def get_sobel_kernels(self):
        Gx_sobel_kernel = np.array([
            [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]
        ])
        Gy_sobel_kernel = np.array([
            [-1, -2, -1],
            [0,  0,  0],
            [1,  2,  1]
        ])
        return Gx_sobel_kernel, Gy_sobel_kernel

    def get_prewitt_kernels(self):
        Gx_prewitt_kernel = np.array([
            [-1,  0,  1],
            [-1,  0,  1],
            [-1,  0,  1]
        ])
        Gy_prewitt_kernel = np.array([
            [-1, -1, -1],
            [0,  0,  0],
            [1,  1,  1]
        ])
        return Gx_prewitt_kernel, Gy_prewitt_kernel

    def get_scharr_kernels(self):
        Gx_scharr_kernel = np.array([
            [-3,  0,   3],
            [-10, 0,  10],
            [-3,  0,   3]
        ])
        Gy_scharr_kernel = np.array([
            [-3, -10, -3],
            [0,   0,  0],
            [3,  10,  3]
        ])
        return Gx_scharr_kernel, Gy_scharr_kernel

    def convolve(self, X, F):
    # height and width of the image
        X_height = X.shape[0]
        X_width = X.shape[1]
    
    # height and width of the filter
        F_height = F.shape[0]
        F_width = F.shape[1]
    
        H = (F_height - 1) // 2
        W = (F_width - 1) // 2
    
        #output numpy matrix with height and width
        out = np.zeros((X_height, X_width))
        #iterate over all the pixel of image X
        for i in np.arange(H, X_height-H):
            for j in np.arange(W, X_width-W):
                sum = 0
                #iterate over the filter
                for k in np.arange(-H, H+1):
                    for l in np.arange(-W, W+1):
                    #get the corresponding value from image and filter
                        a = X[i+k, j+l]
                        w = F[H+k, W+l]
                        sum += (w * a)
                out[i,j] = sum
        #return convolution  
        return out

    def normalize_vector_sobel(self, img, Gx, Gy):
        #normalizing the vectors
        sob_x = self.convolve(img, Gx)
        sob_y = self.convolve(img, Gy)
        return sob_x, sob_y
        
    def normalize_vector_prewitt(self, img, Gx, Gy):
        #normalizing the vectors
        sob_x = self.convolve(img, Gx)
        sob_y = self.convolve(img, Gy)
        return sob_x, sob_y

    def normalize_vector_scharr(self, img, Gx, Gy):
        #normalizing the vectors
        sch_x = self.convolve(img, Gx)
        sch_y = self.convolve(img, Gy)
        return sch_x, sch_y

    def gradient_magnitude(self, x, y):
        #calculate the gradient magnitude of vectors
        out = np.sqrt(np.power(x, 2) + np.power(y, 2))
        # mapping values from 0 to 255
        out = (out / np.max(out)) * 255
        return out

    def output_image(self, out, operator):
        #output images
        cv2.imwrite('output/result.jpg', out)
        plt.imshow(out, cmap = 'gray', interpolation = 'bicubic')
        plt.show()
    
    def use_operators(self, img):
        print("--------------------------------------------------")
        str=input("Qual operador deseja utilizar?\n" +
        "1 - Sobel;\n2 - Prewitt;\n3 - Scharr.\n")
        if (str=='1'):
            Gx, Gy = self.get_sobel_kernels()
            sob_x, sob_y = self.normalize_vector_sobel(img, Gx, Gy)
            out = self.gradient_magnitude(sob_x, sob_y)
            self.output_image(out, str)
        elif (str=='2'):
            Gx, Gy = self.get_prewitt_kernels()
            pre_x, pre_y = self.normalize_vector_prewitt(img, Gx, Gy)
            out = self.gradient_magnitude(pre_x, pre_y)
            self.output_image(out, str)
        elif (str=='3'):
            Gx, Gy = self.get_scharr_kernels()
            sch_x, sch_y = self.normalize_vector_scharr(img, Gx, Gy)
            out = self.gradient_magnitude(sch_x, sch_y)
            self.output_image(out, str)
        else:
            print("O valor inserido é inválido.\n")
        print("--------------------------------------------------")

    def tolerance_up(self, channel):
        min = 255-(255*0.1) <= channel
        max = channel <= 255
        if ((min & max).all()):
            return True
        else:
            return False

    def tolerance_down(self, channel):
        min = 0 <= channel
        max = channel <= 0+(255*0.1)
        if ((min & max).all()):
            return True
        else:
            return False
    
if __name__ == '__main__': 
    img = get_color()
    img_lua, img_chessboard, img02 = img.read_img()
    print("--------------------------------------------------")
    str=input("Em qual imagem deseja aplicar os detectores?\n" +
    "1 - Lua;\n2 - Xadrez;\n3 - Pão.\n")
    if (str=='1'):
        img.use_operators(img_lua)
    elif (str=='2'):
        img.use_operators(img_chessboard)
    elif (str=='3'):
        img.use_operators(img02)
    else:
        print("O valor inserido é inválido.")
    print("--------------------------------------------------")
    
    
    print("**************************************************")
    print("*                      FIM                       *")
    print("**************************************************")
    
