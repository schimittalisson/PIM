from locale import normalize
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpig

class get_color():
    def read_img(self):
        print("**************************************************")
        print("*                     INICIO                     *")
        print("**************************************************")
        str=input("Digite o caminho + nome da imagem\n" +
        "(Ex.:'/home/udesc/Documentos/PIM/gradient/images/Lua1_gray.jpg'):\n")
        img = cv2.imread(str,cv2.IMREAD_GRAYSCALE)
        plt.imshow(image,cmap='gray')
        return img

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

    def get_scharr_kernel(self):
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
        sob_x = self.convolve(img, Gx) / 8.0
        sob_y = self.convolve(img, Gy) / 8.0
        return sob_x, sob_y
        
    def normalize_vector_prewitt(self, img, Gx, Gy):
        #normalizing the vectors
        sob_x = self.convolve(img, Gx) / 6.0
        sob_y = self.convolve(img, Gy) / 6.0
        return sob_x, sob_y

    def gradient_magnitude(self, x, y):
        #calculate the gradient magnitude of vectors
        out = np.sqrt(np.power(x, 2) + np.power(y, 2))
        # mapping values from 0 to 255
        out = (out / np.max(out)) * 255
        return out

    def output_image(self, out):
        #output images
        cv2.imwrite('output/result.jpg', out)
        plt.imshow(out, cmap = 'gray', interpolation = 'bicubic')
        plt.show()
    
    def define_operator(self, img):
        print("--------------------------------------------------")
        str=input("Qual operador deseja utilizar?\n" +
        "1 - Sobel;\n2 - Prewitt;\n3 - Scharr.\n")
        if (str==1):
            Gx, Gy = self.get_sobel_kernels()
            sob_x, sob_y = self.normalize_vector_sobel(img, Gx, Gy)
            out = self.gradient_magnitude(sob_x, sob_y)
            self.output_image(out)
        elif (str==2):
            Gx, Gy = self.get_prewitt_kernels()
            pre_x, pre_y = self.normalize_vector_prewitt(img, Gx, Gy)
            out = self.gradient_magnitude(pre_x, pre_y)
            self.output_image(out)
        elif (str==3):
            print("O carro é azul!")
        else:
            print("O valor inserido é inválido.")
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
    image = img.read_img()
    (blue,green,red)=img.get_img_color_channels(image)
    img.define_color(red, green, blue)
    print("**************************************************")
    print("*                      FIM                       *")
    print("**************************************************")
    
