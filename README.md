# DiffLimAgg

Diffusion-limited aggregation in 2D. Simulate, analyze, visualize.

```python
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
```

This generates sth like the following animation. Video compression kinda messed it up but somehow I like it.

https://user-images.githubusercontent.com/10728380/204252790-652efe63-30d5-4fdf-911f-7627dfc69ff4.mp4


## Install

    pip install ./DiffLimAgg

`DiffLimAgg` was developed and tested for 

* Python 3.6
* Python 3.7
* Python 3.8

So far, the package's functionality was tested on Mac OS X and CentOS only.

## Dependencies

`DiffLimAgg` directly depends on the following packages which will be installed by `pip` during the installation process

* numpy>=1.20
* scipy>=1.9

## License

This project is licensed under the [MIT License](https://github.com/benmaier/DiffLimAgg/blob/main/LICENSE).
Note that this excludes any images/pictures/figures shown here or in the documentation.
