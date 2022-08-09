from abc import abstractclassmethod

class BaseClass(abstractclassmethod):
    def train(self,x,y):
        pass

    def predict(self,x,params):
        pass

    def initialise(self,dims):
        pass