# feb-stack

The team's drivers: every driver that has driven clean in the simulator, one folder per person
under `drivers/`. The car builds this whole repo and launches a driver by name.

```
drivers/
  pranav/
    pranav_driver/      a ROS 2 package with a driver.yaml
    feb_mission/        the packages its launch file needs, listed in driver.yaml under packages:
  akhil/
    ...
```

Nobody edits this repo by hand. `feb submit` in `feb-racing` copies a driver's packages here on
your own branch and opens a pull request; the lead merges it. On the Jetson, `git pull` then
`./feb up --car --driver NAME --mode-name inspection`.

To run someone's merged driver in the simulator on a laptop: `./feb up --ws ~/FEB/feb-stack --driver NAME`.

Experiments stay in your `feb-racing/stack/` on your laptop. `docs/devkit.md` in `feb-racing`
has the whole picture.
