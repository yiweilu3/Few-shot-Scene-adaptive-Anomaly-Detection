from PIL import Image
from torch.utils.data import Dataset,DataLoader

class TrainingDataset(Dataset):
    def __init__(self, img, transform=None):
        self.img= img
        self.transform=transform
    
    # Length of Dataset will be # of Epochs (6)
    def __getitem__(self, index):
        epoch = []
        for task in self.img[index]:
            task_list = []
            for frame_sequence in task:
                frames = []
                gt_path=frame_sequence[3]
                im_path=[]
                img=[]
                for i in range(3):
                    im_path.append(frame_sequence[i])
                for im in im_path:
                    im_opened=Image.open(im).convert('RGB')
                    if self.transform is not None:
                        img.append(self.transform(im_opened))
                gt = Image.open(gt_path).convert('RGB')
                if self.transform is not None:
                    gt = self.transform(gt)
                frames.append(img)
                frames.append(gt)
                task_list.append(frames)
            epoch.append(task_list)
        return epoch

    def __len__(self):
        return len(self.img)
