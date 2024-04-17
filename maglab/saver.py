import time

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

class Saver:
    def __init__(self, name='log.txt'):
        self.name = name
        with open(name, 'w') as f:
            f.write("{}    Start recoding.\n".format(get_time()))
            
    def write(self, content):
        with open(self.name, 'a') as f:
            f.write("{}    {}\n".format(get_time(), content))