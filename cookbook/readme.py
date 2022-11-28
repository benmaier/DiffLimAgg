import numpy as np
import DiffLimAgg as dla
from DiffLimAgg.anim import animate

walkers = dla.Walkers2D(N=50_000,
                    dt=0.01,
                    box=[0,1,0,1],
                    initial_positions='random',
                    position_noise_coefficient=0.0001,
                    velocity_noise_coefficient=.0001,
                    )

exp = dla.Experiment(walkers,
                     walker_radius=0.3*np.sqrt(1/50_000)
                     )

animate(exp)

