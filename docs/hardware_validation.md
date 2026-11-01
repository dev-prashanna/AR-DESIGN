# Future hardware validation procedure (not authorized for execution)

Before power: confirm measured masses, dimensions, joint ordering/directions,
servo model/gearing, electrical/thermal ratings, and limit switches. Replace
provisional dynamics, gains, torque/speed limits and delay with measured values.

Bench-test one joint with the robot supported, an accessible emergency stop,
conservative current/torque and position bounds, and temperature monitoring.
Verify encoder zero and positive directions before testing every joint.

Only after bench review: supported standing in a fall-protection rig; compare
estimated orientation and normalized observations with simulation logs. Keep
an independent watchdog, command timeout, temperature/current shutdown and
operator emergency stop. Then review low-amplitude weight shifts and stepping
in staged sessions. Never use an exported simulation policy directly on the
robot without this separate review and authorization.
