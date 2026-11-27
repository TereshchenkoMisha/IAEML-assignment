"""Clearance-aware reference routes for RL, leaving the teacher's planner untouched.

Routes use static obstacles only. Avoidance of moving obstacles remains the
policy's responsibility. A reference route is guidance, not an action override.
"""
import heapq

import jax
import jax.numpy as jnp
import numpy as np

from env.robotaxi import RobotaxiState


class NavigationState(RobotaxiState):
    reference_path: jax.Array


class ClearancePlanner:
    def __init__(self, obstacles, goal, map_shape, resolution=.25, clearance=1.12):
        self.obstacles = np.asarray(obstacles, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.resolution = resolution
        self.clearance = clearance
        h, w = map_shape
        xs = np.arange(0, w-1+resolution/2, resolution)
        ys = np.arange(0, h-1+resolution/2, resolution)
        self.grid = np.stack(np.meshgrid(xs, ys), axis=-1)
        self.shape = self.grid.shape[:2]
        self.centers = self.grid.reshape(-1, 2)
        distances = np.linalg.norm(self.centers[:,None]-self.obstacles[None], axis=-1).min(1)
        valid = distances >= clearance
        goals = valid & (np.linalg.norm(self.centers-self.goal, axis=-1) < .75)
        if not goals.any():
            raise ValueError('Goal has no clearance-safe approach')
        costs = np.full(len(valid), np.inf)
        self.follow = np.full(len(valid), -1, dtype=int)
        queue = []
        for i in np.flatnonzero(goals):
            costs[i] = 0
            heapq.heappush(queue, (0., int(i)))
        ny, nx = self.shape
        neighbors = [(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if dx or dy]
        while queue:
            cost, index = heapq.heappop(queue)
            if cost > costs[index]:
                continue
            y,x = divmod(index,nx)
            for dy,dx in neighbors:
                yy,xx = y+dy,x+dx
                if not (0<=yy<ny and 0<=xx<nx):
                    continue
                other=yy*nx+xx
                if not valid[other]:
                    continue
                if dx and dy and not (valid[y*nx+xx] and valid[yy*nx+x]):
                    continue  # no diagonal cutting across occupied cells
                step = resolution*np.hypot(dx,dy)
                # Prefer a little more room; hard feasibility remains clearance.
                candidate = cost + step*(1+.15/max(distances[other]-1., .01))
                if candidate < costs[other]:
                    costs[other]=candidate
                    self.follow[other]=index
                    heapq.heappush(queue,(candidate,other))
        self.reachable = np.isfinite(costs)

    def segment_clearance(self, a, b):
        # Exact minimum distance from circle centres to this line segment.
        delta=b-a
        fraction=np.clip(((self.obstacles-a)@delta)/max(delta@delta,1e-12),0,1)
        nearest=a+fraction[:,None]*delta
        return np.linalg.norm(nearest-self.obstacles,axis=-1).min()

    def route(self, start, length=60):
        start=np.asarray(start,dtype=float)
        distance=np.linalg.norm(self.centers-start,axis=-1)
        order=np.argsort(np.where(self.reachable,distance,np.inf))
        first=None
        for index in order[:64]:
            if self.reachable[index] and self.segment_clearance(start,self.centers[index])>=.9999:
                first=int(index)
                break
        if first is None:
            raise ValueError('Cannot connect the start to a clearance-safe route')
        points=[start,self.centers[first]]
        index=first
        for _ in range(len(self.centers)):
            other=self.follow[index]
            if other<0:
                break
            points.append(self.centers[other]);index=other
        else:
            raise RuntimeError('Cycle in Dijkstra predecessor chain')
        # Greedy visibility shortcuts retain the hard clearance bound. The first
        # connector may start at the teacher's exact 1 m boundary.
        if self.segment_clearance(points[0],points[0])<self.clearance:
            simplified=[points[0],points[1]]
            i=1
        else:
            simplified=[points[0]]
            i=0
        while i<len(points)-1:
            threshold=min(self.clearance,self.segment_clearance(points[i],points[i]))
            for j in range(len(points)-1,i,-1):
                if self.segment_clearance(points[i],points[j])>=threshold-1e-6:
                    simplified.append(points[j]);i=j
                    break
            else:
                raise ValueError('Cannot simplify route without crossing an obstacle')
        if len(simplified)>length:
            raise ValueError('Reference path capacity is too small')
        # Insert points, preserving every bend. Uniform resampling can cut corners.
        while len(simplified)<length:
            lengths=np.linalg.norm(np.diff(simplified,axis=0),axis=-1)
            index=int(np.argmax(lengths))
            simplified.insert(index+1,(simplified[index]+simplified[index+1])/2)
        route=np.asarray(simplified,dtype=np.float32)
        if min(self.segment_clearance(a,b) for a,b in zip(route[:-1],route[1:]))<.9998:
            raise ValueError('A reference segment violates the teacher collision radius')
        return route


def attach_navigation(params, states, batched=False):
    """Precompute paths once, outside the JIT loop. Cache the four reflected geometries."""
    cache={}
    def route(state):
        obstacles=np.asarray(state.static_obstacles)
        goal=np.asarray(state.goal_pos)
        identity=(obstacles.tobytes(),goal.tobytes())
        if identity not in cache:
            cache[identity]=ClearancePlanner(obstacles,goal,params.map_height_width)
        return cache[identity].route(np.asarray(state.agent_pos),state.path_array.shape[-2])
    if batched:
        paths=[]
        for i in range(states.agent_pos.shape[0]):
            state=jax.tree_util.tree_map(lambda a:a[i],states)
            paths.append(route(state))
        paths=jnp.asarray(np.stack(paths))
    else:
        paths=jnp.asarray(route(states))
    fields=vars(states).copy()
    fields['path_array']=paths
    return NavigationState(**fields,reference_path=paths)
