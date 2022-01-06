class Acc:
    def __init__(self,name):
        self.name = name
        pass




class Tik:
    none_list=[]
    def __init__(self,stamp):
        self.stamp = stamp
        self.passIn=set()
        self.passOut=set()
        self.passInter=[]

        pass
    def rebuild(self):
        self.__classify()
        self.relevant=[]

    def addPass(self,addPassIn=None,addPassOut=None,addPassInter=None):
        if addPassIn:
            self.passIn|= {addPassIn}
        if addPassOut:

            self.passOut |={addPassOut}

        if addPassInter:
            self.passInter.append(addPassInter)

    def getPass(self,word):
        if word=='In':
            return self.passIn
        elif word=='Out':
            return self.passOut
        elif word =='Inter':
            return self.passInter

    def __classify(self):

        if len(self.passInter) + len(self.passIn) +len(self.passOut) ==1:
            if len(self.passInter):
                self.class_id = 'II'
            else:
                self.class_id = 'I'
        elif len(self.passInter) + len(self.passIn) +len(self.passOut) >1:
            self.class_id = 'III'
        else:
            self.class_id='O'
    def __str__(self):
        ret = "tik stamp:{},tik class:{}".format(self.stamp,self.class_id)
        return ret

    def is_inInter(self,si):
        for inter_set in self.passInter:# 能够应对两个以上的卫星 inter
            if si in inter_set:
                return True
        return False
    def is_in(self,si):
        if (si not in self.passIn )and (si not in self.passOut )and (not self.is_inInter(si)):
            return False
        return True