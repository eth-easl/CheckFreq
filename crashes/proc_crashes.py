import argparse
import time
import logging
import os
import signal
import subprocess

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-4s [%(filename)s] %(message)s',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--interval', type=int, default=30,
                    help='crash interval in seconds')
parser.add_argument('--out', type=str, required=True, help='prefix for stdout files')
parser.add_argument('cmd', nargs=argparse.REMAINDER,
                    help='command to run, comes after --')
args = parser.parse_args()

if len(args.cmd) == 0 or args.cmd[0] != '--':
    raise ValueError(
        'Bad argument for command to run, place it after the -- separator')

cmd = args.cmd[1:]
logger.info("Command to be executed {}".format(cmd))

crash_count = 0
while True:
    outfile = "{}.{}".format(args.out, crash_count)
    outfd = open(outfile, 'w')
    logger.info("Redirecting stdout and stderr to {}".format(outfile))
    proc = subprocess.Popen(cmd, shell=True, stdout=outfd, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(args.interval)
    ret_code = proc.poll()
    if ret_code is not None:
        logger.info("Process terminated with exit code {}".format(ret_code))
        break

    logger.info("Sending SIGKILL to process")
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    outfd.close()
    crash_count += 1
