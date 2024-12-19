class WheelError:
    def __init__(self, errStr):
        self.wrappedErrs = [errStr]
    
    def __str__(self):
        errStr = ""
        for err in reversed(self.wrappedErrs):
            errStr += f"{err}: "
        return errStr.removesuffix(": ")

    def wrap(self, errStr):
        self.wrappedErrs.append(errStr)    