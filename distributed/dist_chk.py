import torch
import os
import enum
import logging
import copy
from collections import OrderedDict
from collections.abc import Mapping
import time

# Checkpointing and restoring, inspired from CheckFreq


class CFCheckpoint(object):

	def __init__(
		self,
		**kwargs):
		self.logger = logging.getLogger(__name__)
		self.tracking_map = OrderedDict()

		for name, ref in kwargs.items():
			if hasattr(ref, 'state_dict'):
				self.tracking_map[name] = ref
			else:
				self.logger.info("Skipping object `{}` in CF Checkpointing. \
				No state_dict() method exposed".format(name))

		self.num_tracking = len(self.tracking_map.keys())

		if self.num_tracking == 0:
			raise ValueError("Nothing to track")

	def __getstate__(self):
		d = self.__dict__.copy()
		if 'logger' in d:
			d['logger'] = d['logger'].name
		return d

	def __setstate__(self, d):
		if 'logger' in d:
			d['logger'] = logging.getLogger(d['logger'])
		self.__dict__.update(d)

	def _snapshot(self, active_snapshot, additional_state=None):
		if active_snapshot == 1:
		# if self.latest_snapshot is not None:
			self.logger.info("Snapshot is not None. Checkpoint underway")
			return False

		self.latest_snapshot = OrderedDict()
		start = time.time()
		# Snapshot the state of tractable items
		for name, ref in self.tracking_map.items():
			if name not in self.latest_snapshot:
				self.latest_snapshot[name] = copy.deepcopy(ref.state_dict())
			else:
				self.logger.info("Repeated entry for {}".format(name))
				return False

		if additional_state:
			self.latest_snapshot.update(additional_state)

		return True

	def _serialize_and_persist(
		self,  
		filepath,
		active_snapshot,
		in_progress_snapshot,
		lock,
		change,
		additional_state=None,
		background=False,
		snapshot_ready=False,
		linkpath=None,
		iter_chk = None,
		epoch_chk = None, 
		overwrite = True):

		while True:

			if background:
				with lock:
					if change.value==0:		
						# print("---------- I am stuck!")
						continue


			print("[{}] START ASYNC".format(time.time()))

			print("------------------------------------------ kwargs: ", background, iter_chk, overwrite)
			if not snapshot_ready:
				self.logger.info("[{}] START SNAPSHOT".format(time.time()))
				start = time.time()
				success = self._snapshot(active_snapshot.value, additional_state=additional_state)
				end1 = time.time()
				print("-------------- Snapshot took: ", end1-start)

				if success:		
					with lock:
						in_progress_snapshot.value = 0
						active_snapshot.value = 1
				else:
					change.value=0
					self.logger.error("Cannot persist. Empty snapshot")
					return
			else:
				with lock:
					if active_snapshot.value == 0:
						change.value=0
						self.logger.error("Cannot persist. Empty snapshot")
						return
			
			snapshot = self.latest_snapshot
			# print(snapshot)
			# Create new stream
			s = torch.cuda.Stream()
			torch.cuda.stream(s)

			# print("Saving : {}".format(filepath))
			torch.save(snapshot, filepath.value)
			# print("Saved : {}".format(filepath))
			# Clear the snapshot.
			with lock:
				active_snapshot.value = 0

			# Ensure its persisted
			f = open(filepath.value, 'a+')
			os.fsync(f.fileno())
			f.close()

			update_stats(filepath.value,overwrite=overwrite,iter_chk=iter_chk)
			print("[{}] END ASYNC".format(time.time()))

			with lock:
				snapshot={}
				change.value=0

			if not background:
				print(" *** ------------------------------------ TEMPORARY, exit now -----------------------------------")
				return	



def update_stats(del_filepath,overwrite = True, iter_chk=None):
        if overwrite:
            if os.path.exists(del_filepath):
                os.remove(del_filepath)
            iter_chk.value = iter_chk

		