#Scheduler for curriculum learning
import math

class CLScheduler:

    def __init__(self, total_steps=40000, num_buckets=None):
        self.total_steps = total_steps
        if num_buckets == None:
            self.start_ratio = 1 / num_buckets # in case of the continuous schedule
        else:
            self.start_ratio = 0
            self.thresholds = [(1 / num_buckets) * k for k in range(1, num_buckets)]

    #pacing functions
    def pf_linear(self, num_steps):
      lamb = self.start_ratio + ((1 - self.start_ratio) / self.total_steps) * num_steps
      lamb = min(lamb, 1)
      return lamb

    def pf_root(self, num_steps, pow):
      lamb = (self.start_ratio**pow + ((1 - self.start_ratio**pow) / self.total_steps) * num_steps) ** (1/pow)
      lamb = min(lamb, 1)
      return lamb

    def pf_geom(self, num_steps):
      eps = 0.001
      lamb = 2 ** ((math.log2(1)-math.log2(self.start_ratio + eps) / self.total_steps) * num_steps + math.log2(self.start_ratio + eps))
      lamb = min(lamb, 1)
      return lamb

    def pf_square(self, num_steps):
      lamb = (num_steps / self.total_steps) ** 2 + self.start_ratio
      lamb = min(lamb, 1)
      return lamb

    def pf_tangent(self, num_steps):
      num_steps -= self.total_steps / 2#parallel shift on x axis
      num_steps *= ((math.pi / 4) / (self.total_steps/2))#normalization
      lamb = ((math.tan(num_steps) + 1) / 2) * ((1-self.start_ratio)/1) + self.start_ratio#parallel shift on y axis
      lamb = min(lamb, 1) 
      return lamb

    def get_schedule(self, schedule_type, pow=2):
        if schedule_type == "linear":
            pf_func = self.pf_linear
        elif schedule_type == "root":
            pf_func = self.pf_root
        elif schedule_type == "geom":
            pf_func = self.pf_geom
        elif schedule_type == "square":
            pf_func = self.pf_square
        elif schedule_type == "tangent":
            pf_func = self.pf_tangent
        x = [i for i in range(1, self.total_steps+1, 1)]
        if schedule_type == "root":
            y = [pf_func(i, pow) for i in x]
        else:
            y = [pf_func(i) for i in x]
        return self.check_transition_steps(y)

    def check_transition_steps(self, y_list):
        transition_steps = [] # the list of steps where a shceduler change the difficulty level of dataset
        for threshold in self.thresholds:
            for x, y in enumerate(y_list):
                if y >= threshold:
                    transition_steps.append(x+1)
                    break
        return transition_steps




#test

s_types = ["linear", "root", "geom", "square"]

#scheduler class
scheduler = CLScheduler(total_steps=300000, num_buckets=3)

for s_type in s_types:
    if s_type == "root":
        for pow in [2, 3]:
            schedule = scheduler.get_schedule(s_type, pow=pow)
            print(f"schedule ({s_type}, {pow}): ", schedule)
    else:
        schedule = scheduler.get_schedule(s_type)
        print(f"schedule ({s_type}): ", schedule)
