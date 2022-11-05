import numpy as np

def get_uniformly_distributed_points(N,box):
    boxx = box[:2]
    boxy = box[2:]
    Lx = np.diff(boxx)[0]
    Ly = np.diff(boxy)[0]
    pos = np.random.rand(N, 2)
    pos[:,0] *= Lx
    pos[:,1] *= Ly
    pos[:,0] -= boxx[0]
    pos[:,1] -= boxy[0]

    return pos

def get_uniformly_distributed_points_on_boundary(N,box):
    boxx = box[:2]
    boxy = box[2:]
    Lx = np.diff(boxx)[0]
    Ly = np.diff(boxy)[0]

    circumference = 2*Lx + 2*Ly
    nx1 = nx2 = int(np.floor(Lx / circumference*N))
    ny1 = ny2 = int(np.floor(Ly / circumference*N))
    counts = [nx1, ny1, nx2, ny2]

    while True:
        diff = N - sum(counts)
        if diff > 0:
            counts[np.random.randint(0,4)] += 1
        else:
            break

    nx1, ny1, nx2, ny2 = counts
    c = counts
    assert(sum(counts) == N)

    pos = np.zeros((N, 2))

    pos[:c[0],0] = np.random.rand(nx1)*Lx + boxx[0]
    pos[:c[0],1] = boxy[0]

    pos[c[0]:sum(c[:2]),0] = boxx[0]
    pos[c[0]:sum(c[:2]),1] = np.random.rand(ny1)*Ly + boxy[0]

    pos[sum(c[:2]):sum(c[:3]),0] = np.random.rand(nx2)*Lx + boxx[0]
    pos[sum(c[:2]):sum(c[:3]),1] = boxy[1]

    pos[sum(c[:3]):sum(c),0] = boxx[1]
    pos[sum(c[:3]):sum(c),1] = np.random.rand(ny2)*Ly + boxy[0]


    return pos

class Walkers2D:

    def __init__(self, N,
                       dt,
                       box=[0,1,0,1],
                       periodic_boundary_conditions=True,
                       initial_positions='random',
                       initial_velocities='zero',
                       position_noise_coefficient=0.0,
                       velocity_noise_coefficient=1.0,
                       force_field=lambda pos: 0.0,
                       drift_field=lambda pos: 0.0,
                       masses=1.0,
                 ):
        assert(N == int(N))
        assert( 0<dt and dt<1)
        self.N = int(N)

        self.boxx = box[:2]
        self.boxy = box[2:]
        self.Lx = np.diff(self.boxx)[0]
        self.Ly = np.diff(self.boxy)[0]

        self.pbc = periodic_boundary_conditions
        self.m = masses
        self.F = force_field
        self.drift = drift_field
        self.Dx = position_noise_coefficient
        self.Dv = velocity_noise_coefficient
        self.dt = dt
        self.facx = np.sqrt(2*self.Dx*dt)
        self.facv = np.sqrt(2*self.Dv*dt)
        if initial_velocities == 'zero':
            self.v = np.zeros((N,2))
        elif initial_velocities.hasattr('shape') and initial_velocities.shape==(N,2):
            self.v = np.array(self.initial_velocities)
        else:
            raise ValueError('Sorry, no idea what to do with `initial_velocities`')

        if initial_positions == 'random':
            self.x = get_uniformly_distributed_points(N, box)
        elif initial_positions == 'boundary':
            self.x = get_uniformly_distributed_points_on_boundary(N, box)
        elif initial_positions == 'mixed':
            nin = N//4
            nout = N - nin
            xin = get_uniformly_distributed_points(nin, box)
            xout = get_uniformly_distributed_points_on_boundary(nout, box)
            self.x = np.vstack((xin, xout))
        elif initial_positions.hasattr('shape') and initial_positions.shape==(N,2):
            self.x = np.array(self.initial_positions)
        else:
            raise ValueError('Sorry, no idea what to do with `initial_positions`')

        self._free_walkers = np.arange(N)

    def fix_first_walker(self):
        # first particle is always in the center and always fixed
        self.x[0,0] = self.boxx[0] + self.Lx/2
        self.x[0,1] = self.boxy[0] + self.Ly/2

        self._free_walkers = np.arange(1,self.N)


    def step(self,free_walkers=None):
        fx = self.facx
        fv = self.facv
        N = self.N
        dt = self.dt
        m = self.m

        if free_walkers is None:
            free_walkers = self._free_walkers
        n = len(free_walkers)

        W = np.random.randn(2*n,2)

        x = self.x[free_walkers,:]
        v = self.v[free_walkers,:]
        F = self.F(x)
        vdr = self.drift(x)

        dx = (v+vdr)*dt + fx * W[:n,:]
        dv = F/m*dt + fv * W[n:,:]

        self.x[free_walkers,:] += dx
        self.v[free_walkers,:] += dv

        self._apply_boundary_conditions()


    def _apply_boundary_conditions(self):
        if not self.pbc:
            raise NotImplementedError('Reflective boundary conditions have not been implemented yet. Sorry.')

        x = self.x
        L = [self.Lx, self.Ly]
        mins = [self.boxx[0],self.boxy[0]]
        maxs = [self.boxx[1],self.boxy[1]]
        for dim in [0,1]:

            while True:
                ndx = np.where(x[:,dim]<mins[dim])[0]
                if len(ndx) == 0:
                    break
                self.x[ndx,dim] += L[dim]

            while True:
                ndx = np.where(x[:,dim]>maxs[dim])[0]
                if len(ndx) == 0:
                    break
                self.x[ndx,dim] -= L[dim]


if __name__ == "__main__":

    walkers = Walkers2D(10,0.1)
    walkers.step()

