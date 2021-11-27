import argparse
import logging
import os
import signal
import subprocess

import numpy as np
from scipy.stats import rv_continuous

logging.basicConfig(format='%(asctime)s %(levelname)-4s [%(filename)s] %(message)s',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--interval', type=int, default=30,
                    help='crash interval in seconds')
parser.add_argument('--stdout', type=str, default='stdout',
                    help='prefix for stdout files')
parser.add_argument('--stderr', type=str, default='stderr',
                    help='prefix for stdout files')
parser.add_argument('--goog', action='store_true')
parser.add_argument('cmd', nargs=argparse.REMAINDER,
                    help='command to run after --')
args = parser.parse_args()

if len(args.cmd) == 0 or args.cmd[0] != '--':
    print('Bad argument for command to run, place it after the -- separator')
    exit(1)

cmd = ' '.join(args.cmd[1:])
logger.info("Command to be executed: {}".format(cmd))


class failure_dist(rv_continuous):
    def __init__(self, t1, t2, b, A):
        super().__init__(a=0)
        self.t1 = t1
        self.t2 = t2
        self.b = b
        self.A = A

    def _cdf(self, t):
        return self.A * (1 - np.exp(-t / self.t1) + np.exp((t - self.b) / self.t2))


crash_count = 0
interval = args.interval
f_fail = failure_dist(t1=1.0, t2=0.8, A=0.5, b=24)

while True:
    if (args.goog):
        interval = f_fail.rvs() / 24 * 3600  # scale to 1hr

    stdoutfile = "{}.{}".format(args.stdout, crash_count)
    stderrfile = "{}.{}".format(args.stderr, crash_count)
    logger.info("Crash {} after {} sec.".format(crash_count, interval))
    logger.info("Redirecting stdout to {}, stderr to {}".format(
        stdoutfile, stderrfile))
    stdoutfd = open(stdoutfile, 'w')
    stderrfd = open(stderrfile, 'w')
    proc = subprocess.Popen(cmd, shell=True, stdout=stdoutfd,
                            stderr=stderrfd, start_new_session=True)
    try:
        ret_code = proc.wait(timeout=interval)
    except subprocess.TimeoutExpired:
        logger.info("Sending SIGKILL to process")
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        crash_count += 1
    except KeyboardInterrupt:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        raise
    else:
        logger.info("Process terminated with exit code {}".format(ret_code))
        break
    finally:
        stdoutfd.close()
        stderrfd.close()
