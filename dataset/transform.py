import torchvision.transforms as transforms
# import data.additional_transforms as add_transforms
# mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] 是ImagneNet提供的普适性正则化参数


class TransformLoader:
    def __init__(self, image_size, normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        # self.jitter_param = jitter_param
    
    def _parse_transform(self, transform_type):
        # if transform_type == 'ImageJitter':
        #     method = add_transforms.ImageJitter(self.jitter_param)
        #     return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type == 'CenterCrop':
            return method(self.image_size) 
        elif transform_type == 'Resize':
            return method([self.image_size, self.image_size]) 
        elif transform_type == 'Scale':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            # RandomHorizontalFlip、ToTensor等无参数的
            return method()

    def get_composed_transform(self, transform_list=["Resize", "ToTensor", "Normalize"]):
        transform_funcs = [self._parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform
