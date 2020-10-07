import redis
import time
import subprocess
import multiprocessing
from multiprocessing import Process, Pipe
import os

def start_flask():
    print('Starting flask')
    subprocess.Popen(['cd extraction && gunicorn --workers 4 --bind 0.0.0.0:5000 wsgi:app'], shell=True)
    time.sleep(10)

def start_redis(redis_db_path):
    args = ['redis-server']
    if redis_db_path:
        print('Starting Redis from', redis_db_path)
        dir = os.path.dirname(redis_db_path)
        if dir:
            if not os.path.exists(dir):
                os.mkdir(dir)
            args.extend(['--dir', dir])
        args.extend(['--dbfilename', os.path.basename(redis_db_path)])
        # args.extend(['--save', '\"\"', '--appendonly', 'no']) # Never Save db
    subprocess.Popen(args)
    time.sleep(10)


def start_openie(install_path):
    print('Starting OpenIE from', install_path)
    subprocess.Popen(['java', '-mx8g', '-cp', '*', \
                      'edu.stanford.nlp.pipeline.StanfordCoreNLPServer', \
                      '-port', '9000', '-timeout', '15000', '-quiet'], cwd=install_path)
    time.sleep(1)


def worker(remote, parent_remote, env, buffer_size, training_type, clear_kg):
    parent_remote.close()
    env.create()
    obs, info, graph_info = env.reset()
    snapshots = [(obs, info, None, env.env.get_state())] * buffer_size
    try:
        done = False
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                if training_type != 'chained' and done:
                    ob, info, graph_info = env.reset()
                    rew = 0
                    done = False
                else:
                    ob, rew, done, info, graph_info = env.step(data)
                if info['valid'] and training_type == 'chained':
                    if clear_kg:
                        snapshots = snapshots[1:] + [(ob, info, None, env.env.get_state())]
                    else:
                        snapshots = snapshots[1:] + [(ob, info, (graph_info.graph_state, graph_info.graph_state_rep), env.env.get_state())]
                remote.send((ob, rew, done, info, graph_info, env.env.get_state()))
            elif cmd == 'load':
                env_str, force_reload, graph_state, ob = data
                if force_reload:
                    env.env.set_state(env_str)
                    graph_info = env.soft_reset(graph_state, ob)
                    remote.send(graph_info)
                else:
                    remote.send(None)
            elif cmd == 'reset':
                ob, info, graph_info = env.reset()
                remote.send((ob, info, graph_info, env.env.get_state()))
            elif cmd == 'get_snapshot':
                remote.send((snapshots))
            elif cmd == 'import_snapshot':
                snapshots = data
            elif cmd == 'clearkg':
                env.clear_kgs()
            elif cmd == 'close':
                env.close()
                break
            elif cmd == 'go_step':
                if done:
                    ob, info, graph_info = env.reset()
                    rew = 0
                    done = False
                else:
                    ob, rew, done, info, graph_info = env.step(data)
                remote.send((ob, rew, done, info, graph_info))
            elif cmd == 'go_reset':
                ob, info, graph_info = env.reset()
                remote.send((ob, info, graph_info))
            elif cmd == 'go_load':
                env_str = data
                env.env.set_state(env_str)
            elif cmd == 'get_str':
                remote.send((env.env.get_state()))
            elif cmd == 'score':
                remote.send((env.env.get_score()))
            elif cmd == 'moves':
                remote.send((env.env.get_moves()))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class VecEnv:

    def __init__(self, num_envs, env, openie_path, redis_db_path, buffer_size, askbert, training_type, clear_kg):
        start_flask()
        start_redis(redis_db_path)
        self.conn_valid = redis.Redis(host='localhost', port=6379, db=0)
        self.closed = False
        self.total_steps = 0
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env, buffer_size, training_type, clear_kg))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        import timeit
        # if self.total_steps % 1024 == 0:
        #     self.conn_valid.flushdb()
        self.total_steps += 1
        self._assert_not_closed()
        assert len(actions) == self.num_envs, "Error: incorrect number of actions."
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return zip(*results)

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        return zip(*results)

    def close_extras(self):
        self.closed = True
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def load_from(self, env_str, mask, graph_state, ob):
        self._assert_not_closed()
        for i, remote in enumerate(self.remotes):
            remote.send(('load', (env_str, mask[i], graph_state, ob)))
        results = [remote.recv() for remote in self.remotes]
        return results
    
    def clear_kgs(self):
        self._assert_not_closed()
        for i, remote in enumerate(self.remotes):
            remote.send(('clearkg', 'xx'))
    
    
    def get_snapshot(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_snapshot', 'xx'))
        results = [remote.recv() for remote in self.remotes]
        return results

    def import_snapshot(self, snapshot):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('import_snapshot', snapshot))

    def go_load_from(self, env_str):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('go_load', env_str))

    def go_step(self, actions):
        if self.total_steps % 1024 == 0:
            self.conn_valid.flushdb()
        self.total_steps += 1
        self._assert_not_closed()
        assert len(actions) == self.num_envs, "Error: incorrect number of actions."
        for remote, action in zip(self.remotes, actions):
            remote.send(('go_step', action))
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return zip(*results)

    def go_reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('go_reset', None))
        results = [remote.recv() for remote in self.remotes]
        return zip(*results)

    def get_env_str(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_str', 'xx'))
        results = [remote.recv() for remote in self.remotes]
        return results

    def get_moves(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('moves', 'xx'))
        results = [remote.recv() for remote in self.remotes]
        return results

    def get_score(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('score', 'xx'))
        results = [remote.recv() for remote in self.remotes]
        return results

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
