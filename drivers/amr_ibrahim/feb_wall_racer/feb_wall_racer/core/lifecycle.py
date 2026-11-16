"""Shared, unchanged supervisor/mission event-order rules."""
import time


class Lifecycle:
    def on_state(self, msg):
        authorized = msg.data.startswith("AUTONOMOUS")
        if not authorized:
            # Stock drive timeout can publish IDLE before our next timer tick.
            # Preserve an already-stale scan fault before dropping live/resetting.
            self._watchdog()
        # Stock no-deadman supervisors can recover/re-arm from zero readiness
        # commands. That edge must not erase a local or supervisor safety stop.
        # Only a fresh operator button session (or a new process) can clear it.
        if msg.data.partition(":")[0].strip() == "STOP" and not self.halted:
            self._controlled_stop("supervisor %s" % msg.data)
        if authorized and self.supervisor_live is not True:
            operator_start = msg.data == "AUTONOMOUS: autonomous button"
            self.autonomous_start_pending = not self.halted or operator_start
        self.supervisor_live = authorized
        if not authorized:
            self.mission_live = False
            self.autonomous_start_pending = False
        self._update_live(msg.data)

    def on_mission_live(self, msg):
        if not msg.data:
            # Mission can observe the same supervisor stop before this node does.
            self._watchdog()
        # A delayed true after STOP/IDLE/MANUAL cannot authorize the next run.
        self.mission_live = bool(msg.data) and self.supervisor_live is not False
        self._update_live("mission live" if msg.data else "mission stopped")

    def _update_live(self, reason):
        live = self.supervisor_live is True and self.mission_live
        if live and not self.live:
            # Mission restarts/heartbeats alone must not clear a latched safety stop.
            if self.halted and not self.autonomous_start_pending:
                return
            new_session = self.autonomous_start_pending
            if new_session:
                self._reset_run()
                self.autonomous_start_pending = False
            self.live = True
            self.live_since = time.monotonic()
            if new_session:
                self.get_logger().info("AUTONOMOUS and mission live: wall map and control state reset; mapping lap first")
            else:
                self.get_logger().info("mission live: resuming the current autonomous session")
        elif self.live and not live:
            self.live = False
            self.live_since = None
            self._command(0.0, 0.0)
            self.get_logger().info("not driving: %s" % reason)

    def _controlled_stop(self, reason):
        if not self.halted:
            self.halted = True
            self.stop_reason = reason
            self.mode = "STOPPED"
            self.get_logger().error("controlled stop: %s" % reason)
        self._command(0.0, 0.0)
