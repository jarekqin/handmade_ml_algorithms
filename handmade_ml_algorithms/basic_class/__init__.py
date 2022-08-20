from abc import ABCMeta,abstractmethod

class BaseClass(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self,x,y):
        pass

    @abstractmethod
    def predict(self,x,params):
        pass

    @abstractmethod
    def initialise(self,dims):
        pass