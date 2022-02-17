
from ast import arg
from cProfile import label
from email.mime import image
from msilib.schema import Dialog
from re import X
import tkinter as tk
from tkinter import Canvas, Frame, filedialog
from turtle import update, width
from main_yolov5 import *
import cv2


def recongnize_img(imgpath):
    net_type='yolov5s'
    try:
        confThreshold=entry_input1.get()
        nmsThreshold=entry_input1.get()
        objThreshold=entry_input1.get()
    except:
        confThreshold=0.5
        nmsThreshold=0.5
        objThreshold=0.5
    yolonet = yolov5(net_type, confThreshold=confThreshold, nmsThreshold=nmsThreshold, objThreshold=objThreshold)
    srcimg = cv2.imread(imgpath)
    dets = yolonet.detect(srcimg)
    srcimg = yolonet.postprocess(srcimg, dets)
    return srcimg

def draw_img(file_path):
   
    img_arr=recongnize_img(imgpath=file_path)
    new_file_path=file_path.split('.')[0][:-1]+'reg.'+'png'
    print(new_file_path)
    cv2.imwrite(new_file_path,img_arr)

    # mycanvas=Canvas(top,width=600,height=480)
    file=tk.PhotoImage(file=new_file_path)
    # image1=mycanvas.create_image(40,0,image=file)
    # mycanvas.grid()
    label_image=tk.Label(top,image=file)
    label_image.grid(row=2)
    top.mainloop()



def click_open():
    file_path=filedialog.askopenfilename(initialdir="/",title="选择图片",
            filetypes=(("image files",".jpg"),("image files","*.png")))
    #print(file_path)
    #return file_path
    draw_img(file_path)


def main_window(top):
    
    top.geometry("640x560")

    label_select=tk.Label(top,text="设置参数:",justify=tk.LEFT)
    label_select.grid(row=0,column=0)
    label_select=tk.Label(top,text="confThresh",justify=tk.LEFT)
    label_select.grid(row=0,column=1)
    entry_input1=tk.Entry(top,justify=tk.RIGHT)
    entry_input1.grid(row=0,column=2)
    label_select=tk.Label(top,text="nmsThresh",justify=tk.LEFT)
    label_select.grid(row=0,column=3)
    entry_input2=tk.Entry(top,justify=tk.RIGHT)
    entry_input2.grid(row=0,column=4)
    label_select=tk.Label(top,text="objThresh",justify=tk.LEFT)
    label_select.grid(row=0,column=5)
    entry_input3=tk.Entry(top,justify=tk.RIGHT)
    entry_input3.grid(row=0,column=6)

    label_select=tk.Label(top,text="选择文件：",justify=tk.LEFT)
    label_select.grid(row=1,column=5)

    button_select=tk.Button(top,text="确定",justify=tk.LEFT,command=click_open)
    button_select.grid(row=1,column=6)

    top.mainloop()

if __name__=="__main__":
    top=tk.Tk()
    main_window(top)




