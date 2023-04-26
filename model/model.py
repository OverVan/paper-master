models = {}


def register(name):
    """ 注册模型、主干网 """
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    model = models[name](**kwargs)
    return model.cuda()
