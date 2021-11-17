import argparse
import time
import logging
import os
import signal
import subprocess

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
parser.add_argument('cmd', nargs=argparse.REMAINDER,
                    help='command to run after --')
args = parser.parse_args()

if len(args.cmd) == 0 or args.cmd[0] != '--':
    print('Bad argument for command to run, place it after the -- separator')
    exit(1)

cmd = ' '.join(args.cmd[1:])
logger.info("Command to be executed {}".format(cmd))

crash_count = 0
while True:
    stdoutfile = "{}.{}".format(args.stdout, crash_count)
    stderrfile = "{}.{}".format(args.stderr, crash_count)
    logger.info("Redirecting stdout to {}, stderr to {}".format(stdoutfile, stderrfile))
    stdoutfd = open(stdoutfile, 'w')
    stderrfd = open(stderrfile, 'w')
    proc = subprocess.Popen(cmd, shell=True, stdout=stdoutfd,
                            stderr=stderrfd, start_new_session=True)
    try:
        time.sleep(args.interval)
    except KeyboardInterrupt:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        raise

    ret_code = proc.poll()
    if ret_code is not None:
        logger.info("Process terminated with exit code {}".format(ret_code))
        break

    logger.info("Sending SIGKILL to process")
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    stdoutfd.close()
    stderrfd.close()
    crash_count += 1
