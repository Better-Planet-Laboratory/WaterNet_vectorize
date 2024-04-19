import numpy as np
import os
from multiprocessing import Process
from time import time as tt, sleep
import psutil
from pathlib import Path
from shutil import rmtree
import warnings


def delete_directory_contents(directory_path: Path):
    """
    Deletes all content in entered directory.
    DO NOT USE THIS FUNCTION IF YOU DON'T KNOW WHAT YOU ARE DOING!
    Parameters
    ----------
    directory_path
    """
    if directory_path.exists() and directory_path.is_dir():
        rmtree(directory_path)
        directory_path.mkdir()
    elif not directory_path.exists():
        warnings.warn(f'Directory {directory_path} does not exist.', category=RuntimeWarning)
    else:
        warnings.warn(f'{directory_path} is not a directory.', category=RuntimeWarning)


def time_elapsed(start: float, spaces: int = 0) -> str:
    """
    Prints the time elapsed from the entered start time.

    Parameters
    ----------
    start - float, start time obtained from time.perf_counter
    """
    end = tt()
    if end - start < 1:
        statement = ' '*spaces + f'Time Elapsed: {end - start:.6f}s'
        print(statement)
    elif end - start < 60:
        statement = ' '*spaces + f'Time Elapsed: {end - start:.2f}s'
        print(statement)
    elif end - start < 3600:
        mins = int((end - start)/60)
        sec = (end - start) % 60
        statement = ' '*spaces + f'Time Elapsed: {mins}m, {sec:.2f}s'
        print(statement)
    else:
        hrs = int((end - start)/3600)
        mins = int(((end - start) % 3600)/60)
        sec = (end - start) % 60
        statement = ' '*spaces + f'Time Elapsed: {hrs}h, {mins}m, {sec:.2f}s'
        print(statement)
    return statement


class SharedMemoryPool:
    def __init__(
            self, func, input_list: list, num_proc: int,
            time_out: float = None,
            sleep_time: float = .1,
            terminate_on_error: bool = False,
            max_memory_usage_percent: float = 75.,
            terminate_memory_usage_percent: float = 90.,
            time_delta_margin: float = 1.,
            use_kwargs: bool = False,
            print_progress: bool = False,
            name: str = None
    ):
        self.func = func
        self.input_list = input_list
        max_proc = os.cpu_count()
        self.time_out = time_out if time_out is not None else np.inf
        if num_proc > max_proc - 1:
            num_proc = max_proc - 1
        elif num_proc <= 0:
            num_proc = min(max(max_proc + num_proc - 1, 1), max_proc - 1)
        self.num_proc = num_proc
        self.current_input_index = 0
        self.process_dict = {}
        self.sleep_time = sleep_time
        self.terminate_on_error = terminate_on_error
        self.max_memory_usage_percent = max_memory_usage_percent
        self.terminate_memory_usage_percent = terminate_memory_usage_percent
        self.time_delta_margin = time_delta_margin
        self.use_kwargs = use_kwargs
        self.to_print_progress = print_progress
        self.name = name
        self.current_input_index = 0
        self.num_completed = 0
        self.num_new_completed = 0
        self.previous_completed = 0
        self.start_time = tt()
        self.num_to_complete = len(self.input_list)

    def has_memory_issues(self):
        current_memory_usage_percent = psutil.virtual_memory().percent
        return current_memory_usage_percent >= self.max_memory_usage_percent

    def has_available_processors(self):
        return len(self.process_dict) < self.num_proc

    def has_more_inputs(self):
        return self.current_input_index < len(self.input_list)

    def get_name(self, current_input_index):
        return f'{self.name}_{current_input_index}' if self.name is not None else None

    def add_new_process(self):
        # print(inputs)
        inputs = self.input_list[self.current_input_index]
        if self.use_kwargs:
            p = Process(target=self.func, kwargs=inputs, name=self.name)
        else:
            p = Process(target=self.func, args=(inputs,), name=self.name)
        p.start()
        self.process_dict[p.pid] = {
            'process': p, 'start_time': tt(), 'inputs': inputs, 'cpu_time': 0, 'cpu_time_delta': 0,
            'init_start_time': tt()
        }
        self.current_input_index += 1

    def fix_memory_usage(self):
        while psutil.virtual_memory().percent > self.terminate_memory_usage_percent:
            pid_to_terminate = list(self.process_dict.keys())[-1]
            newest_start_time = np.inf
            for pid, process_dict in self.process_dict.items():
                if process_dict['init_start_time'] < newest_start_time:
                    pid_to_terminate = pid
                    newest_start_time = process_dict['init_start_time']
            self.terminate_and_restart(pid_to_terminate)

    def check_for_completed_processes_and_timeouts(self):
        self.num_new_completed = 0
        pids = list(self.process_dict)
        for pid in pids:
            process_info_dict = self.process_dict[pid]
            p = process_info_dict['process']
            new_time = cpu_time = process_info_dict['cpu_time']
            start_time = process_info_dict['start_time']
            time_delta = 0
            if not p.is_alive():
                self.remove_process(pid)
                self.num_completed += 1
                self.num_new_completed += 1
            else:
                new_time = sum(psutil.Process(pid).cpu_times())
                time_delta = new_time - cpu_time
            if time_delta > self.time_delta_margin:
                process_info_dict['cpu_time'] = new_time
                process_info_dict['start_time'] = tt()
            else:
                if tt() - start_time > self.time_out:
                    print(f'{pid} timed out')
                    self.terminate_and_restart(pid)

    def terminate_and_restart(self, pid):
        process_info_dict = self.process_dict[pid]
        p = process_info_dict['process']
        p.terminate()
        inputs = process_info_dict['inputs']
        self.input_list.append(inputs)
        self.remove_process(pid)

    def terminate_all(self):
        for process_info_dict in self.process_dict.values():
            q = process_info_dict['process']
            q.terminate()
            q.join()
            q.close()
        raise Exception('One of the processes failed')

    def remove_process(self, pid):
        try:
            p = self.process_dict[pid]['process']
            p.join()
            exitcode = p.exitcode
            p.close()
            self.process_dict.pop(pid)
            if exitcode == 1 and self.terminate_on_error:
                self.terminate_all()
        except KeyError:
            print(pid, self.process_dict.keys())

    def print_progress(self):
        if self.num_new_completed > 0:
            if self.to_print_progress:
                percent_completed = self.num_completed/self.num_to_complete
                ten_percent = (int(100*percent_completed)//10)*10
                if ten_percent > self.previous_completed:
                    print(
                        f'Completed {percent_completed:.2%} ({self.num_completed}/{self.num_to_complete})'
                    )
                    time_elapsed(self.start_time, 2)
                    self.previous_completed = ten_percent

    def run(self):
        while True:
            while self.has_available_processors() and self.has_more_inputs() and not self.has_memory_issues():
                self.add_new_process()
            self.check_for_completed_processes_and_timeouts()
            self.fix_memory_usage()
            self.print_progress()
            if len(self.process_dict) == 0 and self.current_input_index >= len(self.input_list):
                break
            if len(self.process_dict) == self.num_proc or self.has_memory_issues():
                sleep(self.sleep_time)
