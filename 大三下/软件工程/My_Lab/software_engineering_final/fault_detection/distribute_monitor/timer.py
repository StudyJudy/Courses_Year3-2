'''
raft 算法的计时器类型接口抽象
'''

import threading

class Timer:
    def __init__(self, interval, callback, *args):
        self.interval = interval() if callable(interval) else interval
        assert self.interval > 0
        
        self.callback = callback
        self.args = args
        assert callable(self.callback)
        
        self.handler = None
        self.is_active = threading.Lock()
        
    def start(self):
        with self.is_active:
            self.handler = threading.Timer(self.interval, self.callback, self.args)
            self.handler.start()
            
    def stop(self):
        if self.handler is None:
            return
        with self.is_active:
            self.handler.cancel()
            self.handler = None
    
    def reset(self):
        with self.is_active:
            if self.handler is not None:
                self.handler.cancel()
                self.handler = None
            self.handler = threading.Timer(self.interval, self.callback, self.args)
    
    def get_interval(self):
        return self.interval
    
    def __del__(self):
        self.stop()
    

if __name__ == "__main__":
    import time
    begin = time.time()
    def callback(args1, args2):
        global begin
        call_time = time.time()
        print(args1, args2)
        print("callback, {}s".format(call_time - begin))
        begin = call_time
    
    timer = Timer(2, callback, 'xie','zhiying')
    timer.start()
    print('hi')
    time.sleep(5)
    timer.stop()
    print('bye')
    
    