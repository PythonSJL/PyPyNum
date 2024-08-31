class Config:
    """
    Introduction
    ==========
    The Config class is designed to manage and configure application settings. It provides a mechanism to ensure
    consistency in configurations and prevents accidental changes by limiting attribute modification.

    Roadmap
    ==========
    The Config class is currently stable. Future updates may include new configuration options
    to support additional features.

    Usage
    ==========

    Modifying Configuration:
    ----------
    You can modify the configuration by setting attributes directly, but only with predefined boolean values.

    - config.use_latex = True  # Enable LaTeX formatting

    Reading Configuration:
    ----------
    You can read the current configuration by accessing the attributes.

    - print(config.use_unicode)  # Output: False

    Ensuring Exclusive Configuration:
    ----------
    The Config class is designed to be mutually exclusive, meaning only one configuration item can be True at a time.

    - config.use_std = True  # This will automatically set other configuration items to False

    Getting Configuration Information:
    ----------
    Use the `__repr__` method to get a string representation of the current configuration.

    - print(config)  # Output: Config(use_latex=True, use_std=False, use_unicode=False)
    """
    use_latex = False
    use_unicode = False
    use_std = True
    attributes = ("attributes",)

    def __init__(self):
        Config.attributes = tuple(attr for attr in dir(Config) if attr[0] != "_" and attr != "attributes")

    def __setattr__(self, name, value):
        if name in self.attributes:
            value = bool(value)
            if value is True:
                for attr in self.attributes:
                    if attr != name and getattr(self, attr, False) is True:
                        super().__setattr__(attr, False)
            super().__setattr__(name, value)
        else:
            raise AttributeError("Attribute '{}' is read-only and cannot be added".format(name))

    def __delattr__(self, name):
        raise AttributeError("Attribute '{}' cannot be deleted".format(name))

    def __init_subclass__(cls, **kwargs):
        raise NotImplementedError("Config cannot be subclassed")

    def __copy__(self):
        raise TypeError("Config cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("Config cannot be copied")

    def __getattribute__(self, name):
        if name == "__class__":
            raise AttributeError("Access to '__class__' is forbidden")
        return super().__getattribute__(name)

    def __repr__(self):
        return "Config({})".format(", ".join("{}={}".format(attr, getattr(self, attr)) for attr in self.attributes))


config = Config()
del Config
