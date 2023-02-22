import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

image = cv2.imread('/Users/susum/Desktop/horse head_flame.png')
#image = cv2.imread('/Users/susum/Desktop/M31_1.png')
#image = cv2.imread('/Users/susum/Desktop/test10.png')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(11,11),0)
canny=cv2.Canny(blur,30,150,3)
dilated=cv2.dilate(canny,(1,1),iterations=2)
(cnt,heirarchy) = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb,cnt,-1,(0,255,0),3)
print(f'# of stars in the image: {len(cnt)}')
plt.imshow(rgb)
plt.show()
plt.imshow(image)
plt.show()
plt.imshow(gray, cmap='gray')
plt.show()
plt.imshow(blur, cmap='gray')
plt.show()
plt.imshow(canny, cmap='gray')
plt.show()
plt.imshow(dilated, cmap='gray')
plt.show()
"""
contours,_=cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
for cnt in contours:
    rect=cv2.minAreaRect(cnt)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    img=cv2.drawContours(img,[box],0,(0,255,0),3)
plt.imshow(img)
plt.show()
"""
"""
class Apprication(tk.Frame):
    def __init__(self, root=None):
        super().__init__(root,
            width=380,
            height=280,
            borderwidth=1,
            relief='groove')
        self.pack()
        self.pack_propagate(0)
        self.root=root
        self.create_widgets()

    def create_widgets(self):
        quit_btn=tk.Button(self)
        quit_btn['text']='Close'
        quit_btn['command']=self.root.destroy
        quit_btn.pack(side='bottom')

        self.text_box=tk.Entry(self)
        self.text_box['width']=10
        self.text_box.pack()

        submit_btn=tk.Button(self)
        submit_btn['text']='run'
        submit_btn['command']=self.input_handler
        submit_btn.pack()

        self.message=tk.Message(self)
        self.message.pack()

    def input_handler(self):
        text=self.text_box.get()
        self.message['text']=text+'!'

root = tk.Tk()
root.title('Raspberry Garden Telescope')
root.geometry('400x300')
app = Apprication(root=root)
app.mainloop()
"""