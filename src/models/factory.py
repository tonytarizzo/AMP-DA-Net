
import importlib
import json
from .resnet import ResNetSimple
from .simple_cnn import CIFAR10CNN


def _model_factory_from_args(args):
    """
    Returns (ctor, model_tag, cls_name).
    ctor() -> nn.Module for both global and local models.
    model_tag: short string used in slug/meta (e.g. 'resnet', 'cifarcnn', 'custom-mycnn').
    """
    if args.model == 'resnet':
        cls = ResNetSimple
        kwargs = {}
        tag = 'resnet'
        cls_name = 'ResNetSimple'
    elif args.model == 'cifarcnn':
        cls = CIFAR10CNN
        kwargs = {}
        tag = 'cifarcnn'
        cls_name = 'CIFAR10CNN'
    else:
        if not args.custom_model or ':' not in args.custom_model:
            raise ValueError("For --model custom, provide --custom-model like 'pkg.mod:ClassName'")
        mod_name, class_name = args.custom_model.split(':', 1)
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, class_name)
        try:
            kwargs = json.loads(args.custom_kwargs or '{}')
        except json.JSONDecodeError as e:
            raise ValueError(f"--custom-kwargs must be valid JSON: {e}")
        tag = f"custom-{class_name.lower()}"
        cls_name = class_name

    def ctor():
        return cls(**kwargs)

    return ctor, tag, cls_name


def _model_tag_from_args(args):
    if args.model == 'resnet':
        return 'resnet'
    if args.model == 'cifarcnn':
        return 'cifarcnn'
    if args.model == 'custom':
        # Expect --custom-model like 'pkg.mod:ClassName'
        cls = None
        if getattr(args, 'custom_model', '') and ':' in args.custom_model:
            cls = args.custom_model.split(':', 1)[1]
        return f"custom-{(cls or 'model').lower()}"
    return str(args.model)


def build_global_model(args):
    if args.model == "resnet":
        return ResNetSimple()
    if args.model == "cifarcnn":
        return CIFAR10CNN()
    if args.model == "custom":
        mod_str, cls_str = args.custom_model.split(":", 1)
        mod = importlib.import_module(mod_str)
        kwargs = json.loads(args.custom_kwargs or "{}")
        return getattr(mod, cls_str)(**kwargs)
    raise ValueError(f"Unknown --model {args.model!r}")
