import glob
from PIL import Image
import re
import torch.utils.data as data
import torch

def build_label_dicts(root):
  """Build look-up dictionaries for class label, and class description
  Class labels are 0 to 199 in the same order as 
    tiny-imagenet-200/wnids.txt. 
    Class text descriptions are from 
    tiny-imagenet-200/words.txt
  Returns:
    tuple of dicts
      label_dict: 
        keys = synset (e.g. "n01944390")
        values = class integer {0 .. 199}
      class_desc:
        keys = class integer {0 .. 199}
        values = text description from words.txt
  """
  label_dict, class_description = {}, {}
  with open(root+'/data/tiny-imagenet-200/wnids.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset = line[:-1]  # remove \n
      label_dict[synset] = i

  with open(root+'/data/tiny-imagenet-200/words.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset, desc = line.split('\t')
      desc = desc[:-1]  # remove \n
      if synset in label_dict:
        class_description[label_dict[synset]] = desc
    return label_dict, class_description

def load_filenames_labels(mode,root):
  """Gets filenames and labels
  Args:
    mode: 'train' or 'val'
      (Directory structure and file naming different for
      train and val datasets)
  Returns:
    list of tuples: (jpeg filename with path, label)
  """
  label_dict, class_description = build_label_dicts(root)
  filenames_labels = []
  if mode == 'train':
    filenames = glob.glob(root+'/data/tiny-imagenet-200/train/*/images/*.JPEG')
    for filename in filenames:
      match = re.search(r'n\d+', filename)
      label = str(label_dict[match.group()])
      filenames_labels.append((filename, label))
  elif mode == 'val':
    with open(root+'/data/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
      for line in f.readlines():
        split_line = line.split('\t')
        filename = root+'/data/tiny-imagenet-200/val/images/' + split_line[0]
        label = str(label_dict[split_line[1]])
        filenames_labels.append((filename, label))
  return filenames_labels

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

#def get_labeldict(train_dir):
#    classes = [d.name for d in os.scandir(train_dir) if d.is_dir()]
#    classes.sort()
#    class_to_idx = {classes[i]: i for i in range(len(classes))}
#    return class_to_idx
    
class ValData(data.Dataset):
    def __init__(self, list_IDs, labels, transform):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self,index):
        ID = self.list_IDs[index]
        
#        img_dir = self.dir + '/' + 'val_' + str(ID) + '.JPEG'
        img_dir = self.labels[ID][0]
        X = self.transform(default_loader(img_dir))
        y = int(self.labels[ID][1])
        
        return X, y

class TrainData(data.Dataset):
    def __init__(self, list_IDs, labels, transform):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self,index):
        ID = self.list_IDs[index]
        
        img_dir = self.labels[ID][0]
        X = self.transform(default_loader(img_dir))
        y = int(self.labels[ID][1])
        
        return X, y


class AdvData(data.Dataset):
    def __init__(self, list_IDs, labels, data):
        self.labels = labels
        self.list_IDs = list_IDs
        self.data = data
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self,index):
        ID = self.list_IDs[index]
                
        x = torch.from_numpy(self.data[ID])
        y = self.labels[ID]
        
        return x, y