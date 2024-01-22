import json
from lhs import LHSGenerator
import subprocess
import time
import mysql.connector
import os
import numpy as np
from shutil import copyfile
from logger import SingletonLogger
import queue
import pandas as pd
import tensorboard
from torch.utils.tensorboard import SummaryWriter 

class Tuner():
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None):
        self.knobs_config_path = knobs_config_path
        self.knob_nums = knob_nums
        self.knob_idxs = knob_idxs
        self.initialize_knobs()
        self.dbenv = dbenv
        self.bugets = bugets
        self.logger = self.dbenv.logger
    def initialize_knobs(self):
        f = open(self.knobs_config_path)
        knob_tmp = json.load(f)
        KNOB_DETAILS = {}
        if not self.knob_idxs:
            i = 0
            while i < self.knob_nums:
                key = list(knob_tmp.keys())[i]
                KNOB_DETAILS[key] = knob_tmp[key]
                i = i + 1
        else:
            for idx in self.knob_idxs:
                key = list(knob_tmp.keys())[idx]
                KNOB_DETAILS[key] = knob_tmp[key]
        f.close()
        self.knobs_detail = KNOB_DETAILS

class LHSTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets, knob_idxs)
        self.method = "LHS"
    def lhs(self, lhs_num):
            lhs_gen = LHSGenerator(lhs_num, self.knobs_detail)
            lhs_configs = lhs_gen.generate_results()
            return lhs_configs
    def tune(self):
        self.dbenv.step(None)
        knobs_set = self.lhs(self.bugets)
        for knobs in knobs_set:
            self.dbenv.step(knobs)
class GridTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets, knob_idxs=None):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets, knob_idxs)
        self.method = "Grid"

    def _grid_search(self, params_list, results, current_params=None):
        if current_params is None:
            current_params = []
        if not params_list:
            return current_params
        current_dimension = params_list[0]
        for value in current_dimension:
            result = self._grid_search(params_list[1:], results, current_params + [value])
            if result:
                results.append(result)
    
    def sampling(self, interval):
        knobs_list = []
        for knob_name in self.knobs_detail.keys():
            type = self.knobs_detail[knob_name]["type"]
            if type == "integer":
                minv = self.knobs_detail[knob_name]["min"]
                maxv = self.knobs_detail[knob_name]["max"]
                knobs_list.append(list(np.linspace(minv, maxv, interval, dtype=np.int32)))
            else:
                knobs_list.append(self.knobs_detail[knob_name]["enum_values"])
        results = []
        self._grid_search(knobs_list, results)
        return results
    
    def tune(self, interval=10):
        self.dbenv.step(None)
        knobs_set = self.sampling(interval)
        keys = list(self.knobs_detail.keys())
        for rd, ss in enumerate(knobs_set):
            self.logger.info(f"tuning round {rd + 1} begin!!")
            knobs = {}
            for i in range(len(keys)):
                if isinstance(ss[i], np.integer):
                    knobs[keys[i]] = int(ss[i])
                else:
                    knobs[keys[i]] = ss[i]
            self.dbenv.step(knobs)
            self.logger.info(f"tuning round {rd + 1} over!!")
                

class MySQLEnv():
    def __init__(self, host, user, passwd, dbname, workload, objective, method, stress_test_duration, template_cnf_path, real_cnf_path):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.dbname = dbname
        self.workload = workload
        self.objective = objective
        self.method = method
        self.stress_test_duration = stress_test_duration
        self.template_cnf_path = template_cnf_path
        self.real_cnf_path = real_cnf_path

        self.tolerance_time = 20 #seconds
        self._initial()

    def _initial(self):
        self.timestamp = time.time()    
        results_save_dir = f"/home/root3/Tuning/{self.workload}_{self.timestamp}"
        os.mkdir(results_save_dir)
        self.metric_save_path = os.path.join(results_save_dir, f'results_{self.objective}.res')
        self.dbenv_log_path = os.path.join(results_save_dir, 'dbenv.log')
        self.stress_results = os.path.join(results_save_dir, 'stress_results')
        self.stress_logs = os.path.join(results_save_dir, 'stress_logs')
        self.tensorboard_logs = os.path.join("/home/root3/Tuning", 'tb_logs')
        os.mkdir(self.stress_results)
        os.mkdir(self.stress_logs)
        self.logger = SingletonLogger(self.dbenv_log_path).logger
        self.writer = SummaryWriter(log_dir=self.tensorboard_logs, flush_secs=10)
        self.perfs = {}
        self.round = 0
        self.perfs['cur_tps'], self.perfs['default_tps'], self.perfs['best_tps'] = None, None, None
        self.perfs['cur_lat'], self.perfs['default_lat'], self.perfs['best_lat'] = None, None, None

    def _start_mysqld(self):
        proc = subprocess.Popen(['mysqld', '--defaults-file={}'.format(self.real_cnf_path)])
        self.pid = proc.pid
        #print("pid", self.pid)
        count = 0
        start_sucess = True
        self.logger.info('wait for connection')
        time.sleep(1)
        while True:
            try:
                conn = mysql.connector.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.dbname)
                if conn.is_connected():
                    conn.close()
                    self.logger.info('Connected to MySQL database')
                    self.logger.info('mysql is ready!')
                    self.dbsize = self.get_db_size()
                    self.logger.info(f"{self.workload} database size now is {self.dbsize} MB")
                    break
            except Exception as e:
                print(e)

            time.sleep(1)
            count = count + 1
            self.logger.warn("retry connect to DB")
            if count > 600:
                start_sucess = False
                self.logger.error("can not connect to DB")
                break

        return start_sucess
    
    def _kill_mysqld(self):
        mysqladmin = "/home/root3/mysql/bin/mysqladmin"
        sock = "/home/root3/mysql/mysql.sock"
        cmd = '{} -u{} -S {} shutdown'.format(mysqladmin, self.user, sock)
        p_close = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                       close_fds=True)
        try:
            outs, errs = p_close.communicate(timeout=20)
            ret_code = p_close.poll()
            if ret_code == 0:
                self.logger.info("mysqladmin close database successfully")
        except:
            self.logger.warn("force close database by kill -9!!!")
            os.system("ps aux | grep mysqld | grep my.cnf | awk '{print $2}'|xargs kill -9")
        self.logger.info("mysql is shut down")
    
    def get_db_size(self):
        db_conn = mysql.connector.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.dbname)
        sql = 'SELECT CONCAT(round(sum((DATA_LENGTH + index_length) / 1024 / 1024), 2), "MB") as data from information_schema.TABLES where table_schema="{}"'.format(self.dbname)
        cmd = db_conn.cursor()
        cmd.execute(sql)
        res = cmd.fetchall()
        db_size = float(res[0][0][:-2])
        db_conn.close()
        return db_size
    
    def replace_mycnf(self, knobs=None):
        if knobs == None:
            copyfile(self.template_cnf_path, self.real_cnf_path)
            return
        f = open(self.template_cnf_path)
        contents = f.readlines()
        f.close()
        for key in knobs.keys():
            contents.append(f"{key}={knobs[key]}")
        strs = '\n'.join(contents)
        with open(self.real_cnf_path, 'w') as f:
            f.write(strs)
            f.flush()
        self.logger.info("replace mysql cnf file")

    def apply_knobs(self, knobs=None):
        self._kill_mysqld()
        self.replace_mycnf(knobs)
        time.sleep(10)
        success = self._start_mysqld()
        return success
    
    def get_workload_info(self):
        with open("./workloads.json", "r") as f:
            infos = json.load(f)
        if self.workload.startswith("benchbase"):
            infos[self.workload]["cmd"] = infos[self.workload]["cmd"].format(time.time(), self.stress_results, self.stress_logs)
            return infos[self.workload]["cmd"]
        else:
            pass
    
    def parser_metrics(self, path):
        if self.workload.startswith("benchbase"):
            with open(path, "r") as f:
                metrics = json.load(f)
        else:
            pass
        return metrics

    def clean_and_find(self):
        files = os.listdir(self.stress_results)
        if self.workload.startswith("benchbase"):
            info_files = [file for file in files if file.endswith("samples.csv")]
            info_file = sorted(info_files)[-1]
            df = pd.read_csv(os.path.join(self.stress_results, info_file))
            self.tps_std = df["Throughput (requests/second)"].std()
            self.lat_std = df["95th Percentile Latency (microseconds)"].std()
            for file in files:
                if not file.endswith("summary.json"):
                    os.remove(os.path.join(self.stress_results, file))

            files = [file for file in files if file.endswith("summary.json")]
            files = sorted(files)
            return os.path.join(self.stress_results, files[-1])
        else:
            pass


    def get_metrics(self):
        cmd = self.get_workload_info()
        self.logger.info(f"get workload stress test cmd: {cmd}")
        self.logger.info("begin workload stress test")
        p_benchmark = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
        try:
            outs, errs = p_benchmark.communicate(timeout=self.stress_test_duration + self.tolerance_time)
            ret_code = p_benchmark.poll()
            if ret_code == 0:
                self.logger.info("benchmark finished!")
        except Exception as e: 
            self.logger.info(f"{e}")
            return None

        self.logger.info("clean extra files and get metrics file path")
        outfile_path = self.clean_and_find()
        self.logger.info("parser metrics file")
        metrics = self.parser_metrics(outfile_path)
        return metrics

    def step(self, knobs=None):
        self.logger.info(f"round {self.round} begin!!!")
        self.logger.info(f"ready to apply new knobs: {knobs}")
        flag = self.apply_knobs(knobs)
        self.logger.info("apply new knobs success")
        metrics = self.get_metrics()
        if metrics == None:
            self.logger.error("this round stress test fail")
            self.logger.info("round over!!!")
            return
        try:
            if self.workload.startswith("benchbase"):
                metrics["tps_std"] = self.tps_std
                metrics["lat95_std"] = self.lat_std
                metrics['knobs'] = knobs
                metrics['dbsize'] = self.dbsize
                if self.objective == "all":
                    tmp_tps = metrics["Throughput (requests/second)"]
                    tmp_lat = metrics["Latency Distribution"]["95th Percentile Latency (microseconds)"]
                    if not self.perfs['cur_tps']:
                        self.perfs['cur_tps'], self.perfs['default_tps'], self.perfs['best_tps'] = tmp_tps, tmp_tps, tmp_tps
                    else:
                        self.perfs['cur_tps'] = tmp_tps
                        self.perfs['best_tps'] = max(tmp_tps, self.perfs['best_tps'])

                    if not self.perfs['cur_lat']:
                        self.perfs['cur_lat'], self.perfs['default_lat'], self.perfs['best_lat'] = tmp_lat, tmp_lat, tmp_lat
                    else:
                        self.perfs['cur_lat'] = tmp_lat
                        self.perfs['best_lat'] = min(tmp_lat, self.perfs['best_lat'])
                        
                    self.writer.add_scalars(f"tps_{self.workload}_{self.timestamp}_{self.method}" , {'cur': self.perfs['cur_tps'], 'best': self.perfs['best_tps'], 'default': self.perfs['default_tps']}, self.round)
                    self.writer.add_scalars(f"lat_{self.workload}_{self.timestamp}_{self.method}" , {'cur': self.perfs['cur_lat'], 'best': self.perfs['best_lat'], 'default': self.perfs['default_lat']}, self.round)
                else:
                    pass
            else:
                pass
        except Exception as e:
            print(e)
        
        self.save_running_res(metrics)
        self.logger.info(f"save running res to {self.metric_save_path}")
        self.logger.info(f"round {self.round} over!!!")
        self.round += 1

    def save_running_res(self, metrics):
        if self.workload.startswith("benchbase"):
            save_info = json.dumps(metrics)
            with open(self.metric_save_path, 'a+') as f:
                f.write(save_info + '\n')
                f.flush()
        else:
            pass

def grid_tuning_task(knobs_idxs=None):
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'all', 'grid', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    if not knobs_idxs:
        grid_tuner = GridTuner('/home/root3/Tuning/mysql_knobs.json', 2, dbenv, 10)
    else:
        grid_tuner = GridTuner('/home/root3/Tuning/mysql_knobs.json', 2, dbenv, 10, knobs_idxs)
    logger = dbenv.logger
    logger.warn("grid tuning begin!!!")
    grid_tuner.tune()
    logger.warn("grid tuning over!!!")

def lhs_tuning_task():
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_20_16', 'all', 'lhs', 60, '/home/root3/Tuning/template.cnf', '/home/root3/mysql/my.cnf')
    lhs_tuner = LHSTuner('/home/root3/Tuning/mysql_knobs_copy.json', 60, dbenv, 1000)
    logger = dbenv.logger
    logger.warn("lhs tuning begin!!!")
    lhs_tuner.tune()
    logger.warn("lhs tuning over!!!")


class TaskQueue():
    def __init__(self, nums=-1):
        self.queue = queue.Queue(nums)

    def _execute_task(self, task):
        task_func, task_args = task
        task_func(*task_args)
    
    def add(self, task):
        self.queue.put(task)
    
    def run(self):
        while not self.queue.empty():
            task = self.queue.get()
            self._execute_task(task)
    

if __name__ == '__main__':
    # GridSearch
    # pairs = [[0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    # task_queue = TaskQueue()
    # for pair in pairs:
    #     task_queue.add((grid_tuning_task, (pair, )))
    # task_queue.run()
    # LHS
    task_queue = TaskQueue()
    task_queue.add((lhs_tuning_task, ()))
    task_queue.run()