from .m3_api import *


class monitor:
    def __init__(self):
        m3init()
        self.start_time = time.time()
        self.elapsed_minute = -1
        self.cpu_res_x = []
        self.cpu_res_y = []
        self.cpu_err = False
        pass

    def get_elapsed_minute(self):
        return int((time.time() - self.start_time) / 60)

    def get_cpu(self):
        # print('start get_cpu')
        current_time = time.time()
        elapsed_minute = int((current_time - self.start_time) / 60)
        while self.elapsed_minute < elapsed_minute:
            self.elapsed_minute += 1
            self.cpu_res_x, self.cpu_res_y, self.cpu_err = cpu_get_result(self.elapsed_minute, self.cpu_res_x, self.cpu_res_y)
        # print('get_cpu done')
        return self.cpu_res_x, self.cpu_res_y, self.cpu_err

    def get_dram(self):
        return dram_get_result(self.elapsed_minute)


Monitor = monitor()
