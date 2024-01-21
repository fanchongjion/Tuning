import json
from lhs import LHSGenerator
import subprocess
import time
import mysql.connector
import os
from shutil import copyfile

class Tuner():
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets):
        self.knobs_config_path = knobs_config_path
        self.knob_nums = knob_nums
        self.initialize_knobs()
        self.dbenv = dbenv
        self.bugets = bugets
    def initialize_knobs(self):
        f = open(self.knobs_config_path)
        knob_tmp = json.load(f)
        i = 0
        KNOB_DETAILS = {}
        while i < self.knob_nums:
            key = list(knob_tmp.keys())[i]
            KNOB_DETAILS[key] = knob_tmp[key]
            i = i + 1
        KNOBS = list(KNOB_DETAILS.keys())
        f.close()
        self.knobs_detail = KNOB_DETAILS

class LHSTuner(Tuner):
    def __init__(self, knobs_config_path, knob_nums, dbenv, bugets):
        super().__init__(knobs_config_path, knob_nums, dbenv, bugets)
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


class MySQLEnv():
    def __init__(self, host, user, passwd, dbname, workload, objective, stress_test_duration, template_cnf_path, real_cnf_path):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.dbname = dbname
        self.workload = workload
        self.objective = objective
        self.stress_test_duration = stress_test_duration
        self.template_cnf_path = template_cnf_path
        self.real_cnf_path = real_cnf_path

        self.tolerance_time = 20 #seconds
        self._initial()

    def _initial(self):
        self.timestamp = time.time()    
        results_save_dir = f"/home/root3/Tuning/{self.workload}_{self.timestamp}"
        os.mkdir(results_save_dir)
        self.metric_save_path = os.path.join(results_save_dir, 'results.res')
        self.dbenv_log_path = os.path.join(results_save_dir, 'dbenv.log')
        self.stress_results = os.path.join(results_save_dir, 'stress_results')
        self.stress_logs = os.path.join(results_save_dir, 'stress_logs')
        os.mkdir(self.stress_results)
        os.mkdir(self.stress_logs)


    def _start_mysqld(self):
        proc = subprocess.Popen(['mysqld', '--defaults-file={}'.format(self.real_cnf_path)])
        self.pid = proc.pid
        count = 0
        start_sucess = True
        time.sleep(1)
        while True:
            try:
                conn = mysql.connector.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.dbname)
                if conn.is_connected():
                    conn.close()
                    break
            except Exception as e:
                print(e)

            time.sleep(1)
            count = count + 1
            if count > 600:
                start_sucess = False
                break

        return start_sucess
    
    def _kill_mysqld(self):
        #os.system(f"kill -9 {self.pid}")
        os.system("ps aux | grep mysqld | awk '{print $2}'|xargs kill -9")
    
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

    def apply_knobs(self, knobs=None):
        self._kill_mysqld()
        self.replace_mycnf(knobs)
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
        print(cmd)
        p_benchmark = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
        try:
            outs, errs = p_benchmark.communicate(timeout=self.stress_test_duration + self.tolerance_time)
            ret_code = p_benchmark.poll()
            if ret_code == 0:
                print("[{}] benchmark finished!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        except subprocess.TimeoutExpired:
            print("[{}] benchmark timeout!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            pass
        outfile_path = self.clean_and_find()
        metrics = self.parser_metrics(outfile_path)
        return metrics

    def step(self, knobs=None):
        flag = self.apply_knobs(knobs)
        metrics = self.get_metrics()
        try:
            if self.workload.startswith("benchbase"):
                metrics['knobs'] = knobs
            else:
                pass
        except Exception as e:
            print(e)
        
        self.save_running_res(metrics)
    
    def save_running_res(self, metrics):
        if self.workload.startswith("benchbase"):
            save_info = json.dumps(metrics)
            with open(self.metric_save_path, 'w+') as f:
                f.write(save_info + '\n')
        else:
            pass

if __name__ == '__main__':
    dbenv = MySQLEnv('localhost', 'root', '', 'benchbase', 'benchbase_tpcc_200_16', 'tps', 60, 'template.cnf', '/home/root3/mysql/my.cnf')
    print(dbenv.step())
    #lhs_tuner = LHSTuner('./knob_configs/SYSBENCH_shap.json', 60)
    #res = lhs_tuner.tune()
    #print(res)