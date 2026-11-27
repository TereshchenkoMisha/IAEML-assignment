import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from assignment2 import train
from rl.ppo import Config, init_model, save_checkpoint
from rl.protocol import spatial_audit
from rl.select_checkpoint import validation_score
from rl.tasks import make_bank, position_partition
import jax


class ProtocolTests(unittest.TestCase):
    def test_spatial_pools_remain_disjoint_under_all_reflections(self):
        for mid in range(3):
            audit = spatial_audit(mid)
            self.assertTrue(all(n == 0 for n in audit['overlapping_positions'].values()))
            self.assertTrue(all(p['base_candidates'] > 0 for p in audit['pools'].values()))

    def test_strict_banks_exclude_originals_and_use_their_spatial_partition(self):
        for split, expected in [('validation',1),('test',2)]:
            p, bank, original = make_bank(1, 299, size=32, split=split)
            self.assertFalse(np.asarray(original).any())
            self.assertTrue((position_partition(bank.agent_pos,p.map_height_width) == expected).all())

    def test_checkpoint_selection_refuses_test_even_on_training_maps(self):
        groups = [dict(map_id=mid,scenario=scenario,success=1.,collision=0.,split='train_geometry')
                  for mid in [0,1] for scenario in ['original','varied']]
        result = dict(train_maps=[0,1],groups=groups,strict_splits=True,task_split='test')
        with self.assertRaises(ValueError):
            validation_score(result)
        result['task_split'] = 'validation'
        self.assertEqual(validation_score(result),(1.,0.))

    def test_legacy_warm_start_cannot_contaminate_strict_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'legacy.npz'
            save_checkpoint(path,init_model(jax.random.PRNGKey(0),172,128),Config())
            args = argparse.Namespace(seed=1,envs=2,rollout=4,updates=1,minibatch=8,
                bank_size=8,max_steps=60,train_maps=[0,1],navigation='teacher',strict_splits=True,
                log_every=1,checkpoint_every=1,init_checkpoint=path,out=Path(tmp)/'strict')
            with self.assertRaisesRegex(ValueError,'Strict experiment'):
                train(args)
            self.assertFalse(args.out.exists())


if __name__ == '__main__':
    unittest.main()
