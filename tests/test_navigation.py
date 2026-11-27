import unittest
import jax
import jax.numpy as jnp
import numpy as np

from rl.navigation import ClearancePlanner
from rl.rollout import macro_step
from rl.tasks import original_task, make_bank


class NavigationTests(unittest.TestCase):
    def test_all_original_routes_are_collision_free_and_avoid_top_pinch(self):
        for mid in range(3):
            p,s=original_task(mid,navigation='clearance')
            planner=ClearancePlanner(s.static_obstacles,s.goal_pos,p.map_height_width)
            route=np.asarray(s.reference_path)
            distances=[planner.segment_clearance(a,b) for a,b in zip(route[:-1],route[1:])]
            self.assertGreaterEqual(min(distances),.9998)
            self.assertLess(np.linalg.norm(route[-1]-np.asarray(s.goal_pos)),.75)
            if mid==2:
                # Go BELOW the x=10 wall, not through the zero-width top gap.
                crossing=route[(route[:,0]>9.5)&(route[:,0]<10.5)]
                self.assertTrue(len(crossing)>0)
                self.assertLess(float(crossing[:,1].max()),4.1)

    def test_reference_survives_replanning_and_does_not_override_motion(self):
        p,s=original_task(1,navigation='clearance')
        _,plain=original_task(1)
        key=jax.random.PRNGKey(0)
        action=jnp.array([.3,0.])
        run=jax.jit(lambda state:macro_step(key,state,action,p,66))
        _,new,_,_,_=run(s)
        _,base,_,_,_=run(plain)
        np.testing.assert_array_equal(new.path_array,s.reference_path)
        np.testing.assert_allclose(new.agent_pos,base.agent_pos,atol=1e-5)
        self.assertEqual(float(new.speed),float(base.speed))

    def test_reflected_banks_have_safe_reference_routes(self):
        p,bank,_=make_bank(2,45,size=16,navigation='clearance')
        for i in range(16):
            planner=ClearancePlanner(bank.static_obstacles[i],bank.goal_pos[i],p.map_height_width)
            route=np.asarray(bank.reference_path[i])
            self.assertGreaterEqual(min(planner.segment_clearance(a,b)
                for a,b in zip(route[:-1],route[1:])),.9998)


if __name__=='__main__':
    unittest.main()
