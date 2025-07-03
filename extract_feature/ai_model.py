from abc import abstractmethod


class AIModel():
  def __init__(self):
    self.model = None
    self.name = None
    self.name_general = None

  @abstractmethod
  def __call__(self, image_paths, general=False):
    pass