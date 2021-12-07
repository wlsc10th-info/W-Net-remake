import os
from typing import Iterator
import tensorflow as tf

class CheckpointHandler:

    ckpt=tf.train.Checkpoint(step=tf.Variable(0))
    manager = tf.train.CheckpointManager(ckpt, './ckpt', max_to_keep=3)
    
    def save_checkpoint(self):    
        save_path=self.manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def load_checkpoint(self):
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

