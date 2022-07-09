#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt #afficher des courbes
import tensorflow as tf
import numpy as np
from Tkinter import *
import tkFileDialog as filedialog
from PIL import Image, ImageTk
import pyscreenshot as ImageGrab
import random
import os

assert hasattr(tf,"function")

fenetre = Tk() #declaration de la fenetre principale
fenetre.geometry("565x620+680+205") # Definition de la taille et position de la fenetre

class main: 
    def __init__(self,master):
        self.master = master 
        self.color_fg = 'black' #couleur du crayon
        self.color_bg = 'white' #couleur du fond
        self.old_x = None #initialisation a NONE de old_x
        self.old_y = None #initialisation a NONE de old_y
        self.penwidth = 30 #taille du crayon
        self.main() #appelle de la fonction main
        self.c.bind('<B1-Motion>',self.paint) #dessin les trait a l'appuie 
        self.c.bind('<ButtonRelease-1>',self.reset) #stop le dessin au relachement
        self.loaded_model = tf.keras.models.load_model("./Model.h5") #chargement du model
        self.mnist = tf.keras.datasets.mnist 
        (images,label),(image_validation,label_validation) = self.mnist.load_data()
        self.images = images[:10000]

    def paint(self,e):
        if self.old_x and self.old_y: #si old_x!=0 et old_y!=0 alors
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True) #creation d'un ligne de old_x,e.x a old_y,e.y
        self.old_x = e.x #mise a jour de old_x
        self.old_y = e.y #mise a jour de old_y

    def reset(self,e):  
        self.old_x = None #reinitialisation a NONE
        self.old_y = None #reinitialisation a NONE

    def clear(self):
        self.c.delete(ALL) #supprime le contenu du canvas c
        self.canvas.delete(ALL)  #supprime le contenu du canvas 'canvas'
        
    def prediction(self):
        img = self.imgprediction
        img = img.reshape(1,28,28,1) #redimention de img 
        prediction = self.loaded_model.predict(img)[0] #recuperation des prediction faite sur l'image
        prob = max(prediction) #recupere la plus grande prediction
        prediction = np.argmax(prediction) #recupere le label de la plus grande prediction
        txt="Resultat: " + str(prediction) + " (" + str(int(prob*100)) + "%)" #affichage du resultat 
        text=self.canvas.create_text(10,7,text=txt, justify=RIGHT, anchor='nw') #creation du text
        
    def convertimg(self):
        self.imgprediction = self.imgprediction.resize((28,28), Image.ANTIALIAS) #redimention de l'image dessinee/importee
        self.imgprediction = self.imgprediction.convert('L') #convertion en niveau de gris
        self.imgprediction = np.array(self.imgprediction) #convertion en liste
        self.imgprediction = (255-self.imgprediction)/255 #inversion des nuances de gris et normalisation entre 0 et 1 
        self.imgprediction = self.imgprediction.astype('float64') #changement du type en flaot64
        self.prediction() #appelle de la fonction prediction
        
    def imgmnist(self):
        self.clear() #appelle de la fonction clear 
        randomint = random.randint(0,9999) #nombre aleatoire compris entre 0 et 9999
        self.imgprediction = self.images[randomint] #recupère l'image dans le tableau suivant le nombre aleatoire
        plt.axis('off') #enlève les axes de l'image
        plt.imshow(self.imgprediction, cmap="binary") #affiche l'image en niveau de gris
        plt.savefig("imgtmp.png") #sauvegarde l'image
        self.image = Image.open("imgtmp.png") #ouvre l'image avec PIL.Image
        self.image = self.image.resize((560,560)) #redimension de l'image
        self.photo = ImageTk.PhotoImage(self.image) #ouvre l'image avec PhotoImage
        self.c.create_image(0,0,anchor=NW,image=self.photo) #applique l'image sur le canvas
        self.c.pack() #oblige les modification sur le canvas
        os.remove("imgtmp.png") #supprime l'image
        self.prediction() #appelle de la fonction prediction
    
    def charger(self):
        self.clear() #appelle de la fonction clear
        self.filepath = filedialog.askopenfilename(title="Charger une image",filetypes=[('png files','.png'),('all files','.*')])
        self.imgprediction = Image.open(self.filepath)                              #ouvre l'image avec la classe PIL.Image
        self.imgprediction = self.imgprediction.resize((560,560), Image.ANTIALIAS)  #redimensionne l'image
        self.photo = ImageTk.PhotoImage(self.imgprediction)                         #met l'image dans la classe PhotoImage pour pouvoir l'appliquer sur un canvas
        self.c.create_image(0,0,anchor=NW,image=self.photo)                         #applique l'image sur le canvas
        self.c.pack()                                                               #oblige les modification sur le canvas
        self.convertimg()                                                           #appelle de la fonction convertimg
        
    def envoyer(self):
        self.canvas.delete(ALL)                                   #supprime le contenui du canves "canvas"
        x1=2+fenetre.winfo_rootx()+self.c.winfo_x()               #on recupère la position x en haut a gauche du canevas
        y1=2+fenetre.winfo_rooty()+self.c.winfo_y()               #on recupere la position y en haut a gauche du canvas
        x2=x1+560                                                 #on recupère la position x en bas a droite du canevas
        y2=y1+560                                                 #on recupère la position y en bas a droite du canevas
        self.imgprediction = ImageGrab.grab().crop((x1,y1,x2,y2)) #on prend une capture d'image du canvas
        self.convertimg()                                         #appelle de la fonction convertimg
    
    #MENU
    def apropos(self):
        a_fenetre = Tk() #declaration de la fenetre "a propos"
        a_fenetre.title('a PROPOS') #declaration du titre de la fenetre
        #a_fenetre.iconbitmap('logo-icon.ico') #declaration de l'icône de la fenetre
        a_fenetre.geometry("500x110+680+205") #declaration de la taille et de position de la fenetre

        txt1="Ce logiciel a ete developpe par BENADDA Riham et BOUBERT DICK Jayson, dans le cadre de l'unite d'enseignement 'Projet' pour l'universite de Picardie Jules Vernes."
        txt2="Ce logiciel a pour but de reconaître les chiffres ecrit a la main, nous avons essayer de repondre a celui-ci grâce a deux methodes d'entrees differentes, a savoir charger une image contenant un chiffre ou dessiner celui-ci dans une zone prevue a cette effet."
        self.cAP = Canvas(a_fenetre, width=500, height=110, background='white') #declaration d'un canvas
        text=self.cAP.create_text(10,7,text=txt1, justify=LEFT, anchor='nw', width=480) #application de txt1 sur le canvas
        text=self.cAP.create_text(10,60,text=txt2, justify=LEFT, anchor='nw', width=480) #application de txt2 sur le canvas
        self.cAP.pack() #oblige les modification sur le canvas
    
    #MAIN
    def main(self):
        self.c = Canvas(self.master,width=560,height=560,bg='white') #declaration d'un canvas
        self.c.pack(expand=False, side=TOP) #oblige les modification sur le canvas
        
        self.canvas = Canvas(fenetre, width=560, height=25, background='white') #declaration d'un canvas
        self.canvas.pack(side=BOTTOM) #oblige les modification sur le canvas
            
        #MENU
        menubar = Menu(fenetre) #declaration d'une barre de menu
               
        menu0 = Menu(menubar, tearoff=0)  #declaration d'un element de la barre de menu
        menubar.add_command(label="Clear", command=self.clear) #apllication et nommage du boutton
   
        menu1 = Menu(menubar, tearoff=0) #declaration d'un element de la barre de menu
        menubar.add_command(label="Charger une image", command=self.charger) #apllication et nommage du boutton
        
        menu3 = Menu(menubar, tearoff=0) #declaration d'un element de la barre de menu
        menubar.add_command(label="Mnist", command=self.imgmnist) #apllication et nommage du boutton
        
        menu2 = Menu(menubar, tearoff=0) #declaration d'un element de la barre de menu
        menubar.add_command(label="A propos", command=self.apropos) #apllication et nommage du boutton
        
        fenetre.config(menu=menubar) #ajoute le menubar en tant que menu de la fenetre
        
        Button(fenetre, text ='FAIRE UNE PREDICTION', relief=FLAT, bg="#5a6fff", fg="white", activebackground="#4657cc", activeforeground="white", command=self.envoyer, width=79).pack(side=BOTTOM) #ajout d'un boutton prediction sur la fenetre principale
        
main(fenetre) #appelle de la classe main
#fenetre.iconbitmap('logo-icon.ico') #declaration de l'icone 
fenetre.title('BIG BRAINS') #declaration du nom
fenetre.mainloop() #appelle de la fonction mainloop



