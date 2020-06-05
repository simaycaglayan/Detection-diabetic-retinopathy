import os
from PIL import Image

path_dir: str ="D:/DatasetCrop/Train/saglikli"
list_photo = (os.listdir(path_dir))
list_photo.remove('Thumbs.db')
print(list_photo)
num_photo = len(list_photo)
count = 0
numberName_list = []
target_list = []

for j in range(1,num_photo+1):
    numberName_list.append(j)

print(numberName_list)
numberName_str = list(map(str, numberName_list))
print(numberName_str)

for k in numberName_str:
    target_list.append(('D:/DatasetCrop/Train/saglikliresize/'+ k + '.jpg'))

for (i,s) in zip(list_photo, target_list):
        image = Image.open('D:/DatasetCrop/Train/saglikli/'+i)
        new_image = image.resize((400, 400))
        new_image.save(s)
        print(new_image.size)
        count = count + 1

print(count)
print(num_photo)