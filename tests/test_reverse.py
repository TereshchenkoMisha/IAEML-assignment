"""Signed bicycle dynamics and compatibility with forward-only checkpoints."""
import tempfile
import json
import unittest
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env.robotaxi import RobotaxiEnv
from rl.ppo import Config, init_model, load_checkpoint, save_checkpoint
from rl.tasks import make_bank, original_task


class ReverseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = jax.random.PRNGKey(7)
        cls.params, cls.initial = RobotaxiEnv.init_params(cls.key, path_length=30)

    def state(self, speed=0.):
        s = self.initial
        return eqx.tree_at(lambda z: (z.agent_pos, z.goal_pos, z.static_obstacles,
            z.kinematic_obstacles, z.kinematic_obst_velocities, z.path_array,
            z.heading, z.agent_forward_dir, z.speed), s,
            (jnp.array([8.,4.]), jnp.array([18.,4.]), jnp.full_like(s.static_obstacles,100.),
             jnp.full_like(s.kinematic_obstacles,100.), jnp.zeros_like(s.kinematic_obst_velocities),
             jnp.stack([jnp.linspace(8.,18.,30),jnp.full(30,4.)],axis=-1),
             jnp.float32(0),jnp.array([1.,0.]),jnp.float32(speed)))

    def step(self, state, action, params=None):
        return RobotaxiEnv.step(self.key,state,jnp.asarray(action,dtype=jnp.float32),params or self.params)

    def test_reverse_moves_back_without_flipping_vehicle_heading(self):
        _,s,_,done,_ = self.step(self.state(),[-1,0])
        self.assertLess(float(s.speed),0)
        self.assertLess(float(s.agent_pos[0]),8)
        self.assertEqual(float(s.heading),0)
        np.testing.assert_array_equal(s.agent_forward_dir,[1.,0.])
        self.assertFalse(bool(done))

    def test_direction_changes_brake_to_zero_before_reversing(self):
        p = eqx.tree_at(lambda z:(z.fps,z.step_size,z.substeps),self.params,(1,1.,1))
        for initial_speed, command in [(0.1,-1),(-0.1,1)]:
            _,stop,_,_,_ = self.step(self.state(initial_speed),[command,0],p)
            self.assertEqual(float(stop.speed),0)
            _,opposite,_,_,_ = self.step(stop,[command,0],p)
            self.assertGreater(float(opposite.speed)*command,0)

    def test_reverse_uses_propulsion_not_braking_acceleration(self):
        p = eqx.tree_at(lambda z:(z.fps,z.step_size,z.substeps),self.params,(10,.1,1))
        _,s,_,_,_ = self.step(self.state(),[-1,0],p)
        self.assertAlmostEqual(float(s.speed),-(p.max_acceleration-p.rolling_deceleration)*.1,places=6)

    def test_friction_opposes_reverse_motion_and_never_drives_through_zero(self):
        def body(s,_):
            s = self.step(s,[0,0])[1]
            return s,s.speed
        final,speeds = jax.jit(lambda s:jax.lax.scan(body,s,None,length=300))(self.state(-.25))
        speeds=np.asarray(speeds)
        self.assertTrue((speeds<=0).all())
        self.assertTrue((np.diff(speeds)>=-1e-7).all())
        self.assertEqual(float(final.speed),0)

    def test_reverse_steering_has_opposite_yaw_and_stationary_car_does_not_turn(self):
        _,forward,_,_,_ = self.step(self.state(1),[0,1])
        _,reverse,_,_,_ = self.step(self.state(-1),[0,1])
        self.assertGreater(float(forward.heading),0)
        self.assertLess(float(reverse.heading),0)
        self.assertAlmostEqual(float(reverse.heading),-float(forward.heading),places=6)
        _,still,_,_,_ = self.step(self.state(),[0,1])
        self.assertEqual(float(still.heading),0)

    def test_speed_limits_and_signed_observation(self):
        _,s,_,_,_ = self.step(self.state(-self.params.max_reverse_speed),[-1,0])
        self.assertAlmostEqual(float(s.speed),-self.params.max_reverse_speed,places=6)
        vector=RobotaxiEnv.get_observation(s,self.params).as_vector()
        self.assertAlmostEqual(float(vector[0]),-1,places=6)
        self.assertEqual(vector.shape,(43,))
        self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(float(self.step(self.state(self.params.max_speed),[1,0])[1].speed),self.params.max_speed)

    def test_collision_behind_vehicle_and_reset(self):
        s=self.state(-1)
        s=eqx.tree_at(lambda z:z.static_obstacles,s,s.static_obstacles.at[0].set(jnp.array([6.99,4.])))
        _,terminal,_,done,info=self.step(s,[0,0])
        self.assertTrue(bool(done) and bool(info['collision']))
        _,again,reward,done,_=self.step(terminal,[-1,0])
        np.testing.assert_array_equal(again.agent_pos,terminal.agent_pos)
        self.assertEqual(float(reward),0)
        _,reset=RobotaxiEnv.reset(self.key,self.params,self.initial)
        self.assertEqual(float(reset.speed),0)

    def test_jit_vmap_handles_signed_actions(self):
        batch=jax.jit(jax.vmap(lambda a:self.step(self.state(),a)))(jnp.array([[1.,0.],[-1.,0.],[0.,1.]]))
        self.assertGreater(float(batch[1].speed[0]),0)
        self.assertLess(float(batch[1].speed[1]),0)
        self.assertEqual(float(batch[1].speed[2]),0)
        self.assertTrue(np.isfinite(batch[0].as_vector()).all())

    def test_legacy_checkpoint_and_task_helpers_remain_forward_only(self):
        cfg=Config(hidden=16,allow_reverse=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz'
            save_checkpoint(path,init_model(self.key,172,16),cfg)
            _,restored=load_checkpoint(path)
            self.assertTrue(restored.allow_reverse)
            # Reproduce the old public file format without depending on an
            # archived experiment or a particular trained model.
            with np.load(path,allow_pickle=False) as saved:
                arrays={name:saved[name].copy() for name in saved.files}
            config=json.loads(str(arrays['config']))
            config.pop('allow_reverse')
            arrays['config']=np.array(json.dumps(config))
            old_path=Path(tmp)/'legacy.npz'
            np.savez_compressed(old_path,**arrays)
            _,old=load_checkpoint(old_path)
            self.assertFalse(old.allow_reverse)
        p,s=original_task(1)
        self.assertFalse(p.allow_reverse)
        self.assertEqual(float(self.step(s,[-1,0],p)[1].speed),0)
        p,_,_=make_bank(1,1,size=4,allow_reverse=True)
        self.assertTrue(p.allow_reverse)


if __name__=='__main__':
    unittest.main()
