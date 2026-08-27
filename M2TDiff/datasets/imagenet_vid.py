from torch.utils.data import Dataset
class ImageNetVIDDataset(Dataset):
    def __init__(self, root, num_frames=4, transform=None):
        self.root=root; self.num_frames=num_frames; self.transform=transform
        # Add XML parsing / sequence indexing here according to local VID layout.
        self.samples=[]
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        raise NotImplementedError("Implement ImageNet VID sequence indexing and XML annotation parsing for your dataset layout.")
