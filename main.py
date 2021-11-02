from gtts import gTTS
import gtts
from just_playback import Playback
import pygame
import pygame.camera
from pygame.locals import *
import os
from imageai.Classification import ImageClassification
from google_trans_new import google_translator  
import datetime
tts_langs = list(gtts.lang.tts_langs())
translator = google_translator()
execution_path = os.getcwd()
pred = ImageClassification()
pred.setModelTypeAsInceptionV3()
pred.setModelPath(f'{os.getcwd()}\\inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
pred.loadModel()
pygame.camera.init()
camlist = list(pygame.camera.list_cameras())
if len(camlist) > 1:
    print("+=============================+")
    for i in range(len(camlist)):
        a = f"|{i}:{camlist[i]}".ljust(30)+"|"
        print(a)
    print("+=============================+")
    camera = input("Enter Camera number: ")
    while not camera.isnumeric():
        camera = input("Enter Camera number: ")
    while int(camera)>len(camlist)-1:
        camera = input("Enter Camera number: ")
else:
    camera = 0
print("+==============+")
print("|Langs:        |")
for i in range(len(tts_langs)):
    a = f"|{i}:{tts_langs[i]}".ljust(15)+"|"
    print(a)
print("+==============+")
lang = input("Enter language: ")
while not lang.isnumeric():
    lang = input("Enter language: ")
while int(lang) not in range(len(tts_langs)):
    lang = input("Enter language: ")

pygame.init()

pb = Playback()
try:
    os.makedirs("Snaps")
    os.makedirs("outs")
    os.makedirs("sounds")
except OSError:
    pass

screen = pygame.display.set_mode((960, 720))

cam = pygame.camera.Camera(camlist[int(camera)], (1920,1080))
cam.start()

file_num = 0
done_capturing = False

while not done_capturing:
    file_num = file_num + 1
    image = cam.get_image()
    screen.blit(image, (0,0))
    pygame.display.update()

    for f in [ f for f in os.listdir("Snaps/")]:
        os.remove(os.path.join("Snaps/", f))
    # Save every frame
    filename = f"Snaps/{file_num}.png"
    pygame.image.save(image, filename)
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            b = datetime.datetime.now().timestamp()
            image3 = cam.get_image()
            a = file_num
            pygame.image.save(image3, f'outs/{a}.png')
            predictions, probabilities = pred.classifyImage(f"outs/{a}.png", result_count=1)
            text= translator.translate(str(predictions[0]).replace("_", " "),lang_tgt=tts_langs[int(lang)])
#            print(text)
            #language='en-US'
            speech=gTTS(text=text,lang=tts_langs[int(lang)],slow=False)
            speech.save(f"sounds/sound{a}.mp3")
            pb.stop()
            pb.load_file(f'sounds/sound{a}.mp3')
            pb.play()
            c = datetime.datetime.now().timestamp()
            print(c-b)
        elif event.type == pygame.QUIT:
            pb.stop()
            done_capturing = True
            filelist = [ f for f in os.listdir("Snaps/")]
            for f in filelist:
                os.remove(os.path.join("Snaps/", f))
            filelist = [ f for f in os.listdir("sounds/")]
            for f in filelist:
                os.remove(os.path.join("sounds/", f))
            filelist = [ f for f in os.listdir("outs/")]
            for f in filelist:
                os.remove(os.path.join("outs/", f))
            exit()


os.system("avconv -r 8 -f image2 -i Snaps/%04d.png -y -qscale 0 -s 1920x1080 -aspect 4:3 result.avi")
