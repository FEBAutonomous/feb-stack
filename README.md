# feb-stack

The team's drivers: every driver that has driven clean in the simulator, one folder per person
under `drivers/`. The car builds this whole repo and launches a driver by name.

```
drivers/
  pranav/
    pranav_driver/      a ROS 2 package with a driver.yaml; packages: lists the ones that are yours
  akhil/
    ...
shared/
  feb_mission/          the supervisor's state as go/stop hooks
  feb_cone_ordering/    the team's cone ordering service
```

`shared/` is one copy of the packages every driver may use, kept by the lead; a driver's
`packages:` lists only its own, and `feb submit` refuses a name that is already in `shared/`.

Nobody edits this repo by hand. `feb submit` in `feb-racing` copies a driver's packages here on
your own branch and opens a pull request; the lead merges it. On the Jetson, `git pull` then
`./feb up --car --driver NAME --mode-name inspection`.

To run someone's merged driver in the simulator on a laptop: `./feb up --ws ~/FEB/feb-stack --driver NAME`.

Experiments stay in your `feb-racing/stack/` on your laptop. `docs/devkit.md` in `feb-racing`
has the whole picture.
