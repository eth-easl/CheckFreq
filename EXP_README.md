## Experiment with CheckFreq functionality

To experiment with CheckFreq optimization set the following flags with the MANUAL configuration:
* --at_gpu: To do a snapshot of the model in GPU (default: False)
* --noop: No optimization (synchronous checkpointing)
* --pipeio: Pipeline only the "persist" part. The snapshot will be done synchronously. This is enabled only if the snapshot is done in CPU.

