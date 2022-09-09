import numpy as np
import cv2

class get_color():
    def read_img(self):
        print("**************************************************")
        print("*                     INICIO                     *")
        print("**************************************************")
        str=input("Digite o caminho + nome da imagem\n" +
        "(Ex.:'/home/udesc/Documentos/pim/images/red.png'):\n")
        img=cv2.imread(str)
        return img

    def get_img_color_channels(self, img):
        b,g,r = cv2.split(img)
        
        return b,g,r

    def define_color(self, red, green, blue):
        print("--------------------------------------------------")
        print("Cor do carro:")
        if(self.tolerance_up(np.mean(red)) and self.tolerance_down(np.mean(green)) and self.tolerance_down(np.mean(blue))):
            print("O carro é vermelho!")
        elif (self.tolerance_up(np.mean(green)) and self.tolerance_down(np.mean(red)) and self.tolerance_down(np.mean(blue))):
            print("O carro é verde!")
        elif (self.tolerance_up(np.mean(blue)) and self.tolerance_down(np.mean(red)) and self.tolerance_down(np.mean(green))):
            print("O carro é azul!")
        elif (self.tolerance_down(np.mean(red)) and self.tolerance_down(np.mean(green)) and self.tolerance_down(np.mean(blue))):
            print("O carro é preto!")
        elif (self.tolerance_up(np.mean(red)) and self.tolerance_up(np.mean(green)) and self.tolerance_up(np.mean(blue))):
            print("O carro é branco!")
        else:
            print("A cor do carro está fora dos padrões aceitos.")
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
    
