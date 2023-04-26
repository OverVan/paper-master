datasets = {}


def register(name):
    """ 注册数据集 """
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    dataset = datasets[name](**kwargs)
    return dataset
